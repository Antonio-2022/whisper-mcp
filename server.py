"""Local-only Whisper MCP server with diarization, chunking, and language detection.

Design:
- Bound to 127.0.0.1 only — never reachable from the network.
- Lazy-loads mlx_whisper on first call; auto-unloads after IDLE_UNLOAD_S seconds.
- Long transcriptions run as background jobs (start_transcribe + get_transcribe).
- For audio > CHUNK_DURATION_S: splits via ffmpeg, transcribes each chunk, merges.
- Language auto-detection: 30-second sample pass when language=None.
- Speaker diarization via pyannote.audio 3.x (requires HF_TOKEN env var).
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# ---- constants ---------------------------------------------------------------

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
IDLE_UNLOAD_S = 600       # drop mlx model after 10 min idle
CHUNK_DURATION_S = 1800   # split audio longer than 30 min into chunks
LANG_DETECT_SAMPLE_S = 30 # seconds of audio used for language detection

# ---- logging -----------------------------------------------------------------

LOG = logging.getLogger("whisper-mcp")
LOG.setLevel(logging.DEBUG)

_log_file = Path.home() / "Library" / "Logs" / "whisper-mcp.log"
_fh = logging.handlers.RotatingFileHandler(
    _log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))
LOG.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(_sh)

# ---- model lifecycle ---------------------------------------------------------

_model_lock = threading.Lock()   # single MPS lock: serializes both mlx_whisper and pyannote
_last_used: float = 0.0
_model_loaded: bool = False
_watchdog_started: bool = False


def _unload_model() -> None:
    global _model_loaded
    if not _model_loaded:
        return
    try:
        import mlx_whisper.load_models as lm
        if hasattr(lm, "_models"):
            lm._models.clear()
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception as exc:
        LOG.warning("model unload failed: %s", exc)
    gc.collect()
    _model_loaded = False
    LOG.info("mlx_whisper model unloaded (idle timeout)")


async def _idle_watchdog() -> None:
    while True:
        await asyncio.sleep(60)
        if _model_loaded and (time.monotonic() - _last_used) > IDLE_UNLOAD_S:
            _unload_model()


def _ensure_watchdog() -> None:
    global _watchdog_started
    if _watchdog_started:
        return
    try:
        asyncio.get_running_loop().create_task(_idle_watchdog())
        _watchdog_started = True
    except RuntimeError:
        pass


# ---- audio helpers -----------------------------------------------------------

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
FFMPEG  = shutil.which("ffmpeg")  or "/opt/homebrew/bin/ffmpeg"


def _get_duration_s(path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    data = json.loads(r.stdout)
    for stream in data.get("streams", []):
        dur = stream.get("duration")
        if dur:
            return float(dur)
    raise RuntimeError(f"ffprobe: no duration found in {path}")


def _split_audio(path: str, chunk_s: int) -> tuple[list[tuple[str, float]], str]:
    """Split audio into 16kHz mono WAV chunks. Returns ([(chunk_path, offset_s)], tmpdir)."""
    duration = _get_duration_s(path)
    tmpdir = tempfile.mkdtemp(prefix="whisper-mcp-chunks-")
    chunks: list[tuple[str, float]] = []
    offset = 0.0
    idx = 0
    while offset < duration:
        chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.wav")
        r = subprocess.run(
            [
                FFMPEG, "-y", "-i", path,
                "-ss", str(offset), "-t", str(chunk_s),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                chunk_path,
            ],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(
                f"ffmpeg chunk failed at offset {offset}s: {r.stderr.decode()[:300]}"
            )
        chunks.append((chunk_path, offset))
        offset += chunk_s
        idx += 1
    LOG.debug("split %s into %d chunks (%ds each)", path, len(chunks), chunk_s)
    return chunks, tmpdir


def _detect_language_from_sample(path: str) -> str:
    """Run a short sample through mlx_whisper for language detection. Call inside _model_lock."""
    import mlx_whisper

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", path, "-t", str(LANG_DETECT_SAMPLE_S),
             "-ar", "16000", "-ac", "1", tmp_path],
            capture_output=True, timeout=30,
        )
        out = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=MODEL_REPO,
            language=None,
            task="transcribe",
        )
        return out.get("language") or "en"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---- diarization helpers -----------------------------------------------------

_diarize_pipeline = None
_diarize_pipeline_lock = threading.Lock()


def _patch_pyannote_compat() -> None:
    """Apply two compatibility patches needed for pyannote.audio 3.x with recent deps:

    1. huggingface_hub >=1.0 dropped the `use_auth_token` kwarg from hf_hub_download.
       Pyannote 3.4 still passes it. We translate it to `token=` transparently.

    2. torch >=2.6 changed `weights_only` default to True, which breaks pyannote's
       checkpoint loading (its .ckpt files contain custom Python objects).
       We patch lightning_fabric / pytorch_lightning's _load to use weights_only=False.
    """
    import sys
    import functools
    import torch
    import huggingface_hub
    from huggingface_hub import hf_hub_download as _orig_dl

    # --- Patch 1: hf_hub_download use_auth_token → token ---
    if not getattr(_orig_dl, "_pyannote_compat_patched", False):
        @functools.wraps(_orig_dl)
        def _compat_dl(*args, use_auth_token=None, **kwargs):
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return _orig_dl(*args, **kwargs)
        _compat_dl._pyannote_compat_patched = True  # type: ignore[attr-defined]
        huggingface_hub.hf_hub_download = _compat_dl
        for mod in list(sys.modules.values()):
            try:
                if mod is not None and getattr(mod, "hf_hub_download", None) is _orig_dl:
                    mod.hf_hub_download = _compat_dl
            except Exception:
                pass
        LOG.debug("patch 1 applied: hf_hub_download use_auth_token compat")

    # --- Patch 2: torch.load weights_only=False for pyannote checkpoints ---
    def _compat_load(path_or_url, map_location=None, weights_only=None):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)

    try:
        import lightning_fabric.utilities.cloud_io as _lf
        if not getattr(_lf._load, "_pyannote_compat_patched", False):
            _lf._load = _compat_load
            LOG.debug("patch 2a applied: lightning_fabric._load weights_only compat")
    except Exception:
        pass

    try:
        import pytorch_lightning.core.saving as _pl_saving
        if not getattr(_pl_saving.pl_load, "_pyannote_compat_patched", False):
            _pl_saving.pl_load = _compat_load
            LOG.debug("patch 2b applied: pytorch_lightning pl_load weights_only compat")
    except Exception:
        pass


def _load_diarize_pipeline():
    """Load (or return cached) pyannote diarization pipeline."""
    global _diarize_pipeline
    with _diarize_pipeline_lock:
        if _diarize_pipeline is not None:
            return _diarize_pipeline
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Set it in the launchd plist or ~/.config/whisper-mcp/.env "
                "to enable speaker diarization."
            )
        try:
            _patch_pyannote_compat()
            from pyannote.audio import Pipeline
            import torch
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
            pipeline = pipeline.to(device)
            _diarize_pipeline = pipeline
            LOG.info("pyannote diarization pipeline loaded on %s", device)
            return pipeline
        except Exception as exc:
            raise RuntimeError(f"Failed to load diarization pipeline: {exc}") from exc


def _diarize_audio(path: str, num_speakers: int | None) -> list[dict[str, Any]]:
    """Run speaker diarization. Returns [{start, end, speaker}].

    Pyannote requires WAV input — converts the source file via ffmpeg if needed.
    """
    pipeline = _load_diarize_pipeline()

    tmp_wav_path: str | None = None
    if not path.lower().endswith(".wav"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="whisper-mcp-diarize-")
        tmp_wav_path = tmp.name
        tmp.close()
        LOG.info("diarize: converting %s to 16kHz mono WAV for pyannote...", path)
        r = subprocess.run(
            [FFMPEG, "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_wav_path],
            capture_output=True, timeout=600,
        )
        if r.returncode != 0:
            os.unlink(tmp_wav_path)
            raise RuntimeError(f"ffmpeg WAV conversion failed: {r.stderr.decode()[:300]}")
        diarize_path = tmp_wav_path
    else:
        diarize_path = path

    try:
        kwargs: dict[str, Any] = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        with _model_lock:
            diarization = pipeline(diarize_path, **kwargs)
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        LOG.debug("diarization: %d speaker turns from %s", len(segments), path)
        return segments
    finally:
        if tmp_wav_path:
            try:
                os.unlink(tmp_wav_path)
            except OSError:
                pass


def _assign_speakers(
    transcription_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label each transcription segment with the speaker covering its midpoint."""
    for seg in transcription_segments:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = "UNKNOWN"
        matched = False
        for turn in speaker_turns:
            if turn["start"] <= mid <= turn["end"]:
                speaker = turn["speaker"]
                matched = True
                break
        if not matched and speaker_turns:
            # Fallback: nearest turn boundary
            speaker = min(
                speaker_turns,
                key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)),
            )["speaker"]
        seg["speaker"] = speaker
    return transcription_segments


def _format_transcript_with_speakers(segments: list[dict[str, Any]]) -> str:
    """Group consecutive same-speaker segments into labeled blocks."""
    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer and current_speaker is not None:
            lines.append(f"\n[{current_speaker}]")
            lines.append(" ".join(buffer))

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            buffer = [text]
        else:
            buffer.append(text)
    flush()
    return "\n".join(lines).strip()


# ---- job store ---------------------------------------------------------------

@dataclass
class Job:
    id: str
    path: str
    language: str | None
    task: str
    diarize: bool = False
    num_speakers: int | None = None
    status: str = "queued"           # queued | running | done | error
    started_at: float = 0.0
    finished_at: float = 0.0
    text: str | None = None
    transcript_with_speakers: str | None = None
    detected_language: str | None = None
    duration_s: float = 0.0
    segment_count: int = 0
    speakers: list[str] | None = None
    error: str | None = None
    raw_segments: list[dict] | None = field(default=None, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        elapsed = (
            ((self.finished_at or time.monotonic()) - self.started_at)
            if self.started_at else 0.0
        )
        d: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "elapsed_s": round(elapsed, 1),
            "path": self.path,
        }
        if self.status == "done":
            d.update(
                text=self.text,
                language=self.detected_language,
                audio_duration_s=round(self.duration_s, 1),
                segments=self.segment_count,
            )
            if self.transcript_with_speakers:
                d["transcript_with_speakers"] = self.transcript_with_speakers
            if self.speakers:
                d["speakers"] = self.speakers
        elif self.status == "error":
            d["error"] = self.error
        return d


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_S = 3600


def _gc_jobs() -> None:
    now = time.monotonic()
    with _JOBS_LOCK:
        stale = [
            jid for jid, j in _JOBS.items()
            if j.status in {"done", "error"} and (now - j.finished_at) > _JOB_TTL_S
        ]
        for jid in stale:
            del _JOBS[jid]


# ---- job runner --------------------------------------------------------------

def _run_job(job: Job) -> None:
    global _last_used, _model_loaded
    tmpdir: str | None = None
    try:
        job.status = "running"
        job.started_at = time.monotonic()
        LOG.info(
            "job %s start — path=%s lang=%s task=%s diarize=%s",
            job.id, job.path, job.language, job.task, job.diarize,
        )

        # Phase 1: Transcription (inside model lock — mlx is single-GPU)
        all_segments: list[dict[str, Any]] = []
        detected_language: str | None = None

        with _model_lock:
            import mlx_whisper

            language = job.language

            # Auto-detect language from a short sample before full transcription
            if language is None:
                LOG.info(
                    "job %s: auto-detecting language from %ds sample...",
                    job.id, LANG_DETECT_SAMPLE_S,
                )
                language = _detect_language_from_sample(job.path)
                LOG.info("job %s: detected language=%s", job.id, language)

            duration_s = _get_duration_s(job.path)
            job.duration_s = duration_s
            LOG.info(
                "job %s: audio duration=%.1fs, chunked=%s",
                job.id, duration_s, duration_s > CHUNK_DURATION_S,
            )

            if duration_s > CHUNK_DURATION_S:
                chunks, tmpdir = _split_audio(job.path, CHUNK_DURATION_S)
                for chunk_path, offset_s in chunks:
                    LOG.info(
                        "job %s: transcribing chunk offset=%.0fs", job.id, offset_s
                    )
                    out = mlx_whisper.transcribe(
                        chunk_path,
                        path_or_hf_repo=MODEL_REPO,
                        language=language,
                        task=job.task,
                        word_timestamps=True,
                    )
                    for seg in out.get("segments", []):
                        seg["start"] += offset_s
                        seg["end"] += offset_s
                        for word in seg.get("words", []):
                            word["start"] += offset_s
                            word["end"] += offset_s
                        all_segments.append(seg)
                    if not detected_language:
                        detected_language = out.get("language")
            else:
                out = mlx_whisper.transcribe(
                    job.path,
                    path_or_hf_repo=MODEL_REPO,
                    language=language,
                    task=job.task,
                    word_timestamps=True,
                )
                all_segments = out.get("segments", [])
                detected_language = out.get("language")

            _model_loaded = True
            _last_used = time.monotonic()

        job.detected_language = detected_language or language
        LOG.info(
            "job %s: transcription done — %d segments, lang=%s",
            job.id, len(all_segments), job.detected_language,
        )

        # Phase 2: Speaker diarization (outside model lock — separate model)
        if job.diarize:
            LOG.info("job %s: starting speaker diarization...", job.id)
            try:
                speaker_turns = _diarize_audio(job.path, job.num_speakers)
                all_segments = _assign_speakers(all_segments, speaker_turns)
                job.speakers = sorted({t["speaker"] for t in speaker_turns})
                LOG.info(
                    "job %s: diarization done — %d speakers: %s",
                    job.id, len(job.speakers), job.speakers,
                )
            except Exception as diarize_exc:
                LOG.error(
                    "job %s: diarization failed (plain transcript still available): %s",
                    job.id, diarize_exc,
                )
                # Surface as a warning in the result, not a hard failure
                job.error = (
                    f"diarization failed: {diarize_exc}; "
                    "plain transcript is still available in 'text'"
                )

        # Phase 3: Build output
        job.raw_segments = [
            {k: v for k, v in s.items() if k != "words"}  # drop word-level timestamps to save memory
            for s in all_segments
        ]
        job.text = " ".join(s.get("text", "").strip() for s in all_segments).strip()
        job.segment_count = len(all_segments)

        if job.diarize and all_segments and "speaker" in all_segments[0]:
            job.transcript_with_speakers = _format_transcript_with_speakers(all_segments)

        job.status = "done"
        job.finished_at = time.monotonic()
        LOG.info(
            "job %s done in %.1fs — %d segs, audio=%.1fs, lang=%s, speakers=%s",
            job.id,
            job.finished_at - job.started_at,
            job.segment_count,
            job.duration_s,
            job.detected_language,
            job.speakers,
        )

    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.finished_at = time.monotonic()
        LOG.error("job %s failed: %s\n%s", job.id, job.error, traceback.format_exc())
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---- path validation ---------------------------------------------------------

def _validate_path(path: str) -> Path | dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        return {"error": f"path must be absolute, got: {path}"}
    if not p.exists() or not p.is_file():
        return {"error": f"file not found: {p}"}
    return p


# ---- MCP tools ---------------------------------------------------------------

mcp = FastMCP("whisper")


@mcp.tool
def start_transcribe(
    path: str,
    language: str | None = None,
    task: str = "transcribe",
    diarize: bool = False,
    num_speakers: int | None = None,
) -> dict[str, Any]:
    """Start a background transcription job. Returns a job_id immediately.

    Use get_transcribe(job_id) to poll. Always returns fast regardless of audio length.

    Args:
        path: Absolute path to a local audio file (m4a, mp3, wav, mp4, opus, flac…).
        language: BCP-47 language code (e.g. "ru", "de", "en") or full name ("Russian").
            Omit to auto-detect via a 30-second sample — always provide if known to avoid
            transcription quality issues from incorrect language detection.
        task: "transcribe" (same language output) or "translate" (output in English).
        diarize: True to identify and label speakers. Requires HF_TOKEN env var and
            accepted terms at huggingface.co/pyannote/speaker-diarization-3.1.
        num_speakers: Known speaker count — improves diarization accuracy when provided.
    """
    _ensure_watchdog()
    _gc_jobs()
    p = _validate_path(path)
    if isinstance(p, dict):
        return p

    job = Job(
        id=uuid.uuid4().hex[:12],
        path=str(p),
        language=language,
        task=task,
        diarize=diarize,
        num_speakers=num_speakers,
    )
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    job.thread = threading.Thread(
        target=_run_job, args=(job,), name=f"whisper-{job.id}", daemon=True
    )
    job.thread.start()
    LOG.info("job %s queued: %s", job.id, str(p))
    return job.public()


@mcp.tool
def get_transcribe(job_id: str) -> dict[str, Any]:
    """Poll a transcription job by id. Returns status and (when done) the transcript.

    When status is 'done', the result contains:
      - text: plain transcript (no speaker labels)
      - transcript_with_speakers: speaker-labeled blocks (present when diarize=True)
      - speakers: list of speaker IDs (e.g. ["SPEAKER_00", "SPEAKER_01"])
      - language: detected/used language code
      - audio_duration_s: total audio length in seconds
      - segments: number of transcription segments

    Typical completion times (warm model / cold model):
      - 10-min audio: ~30s / ~60s
      - 30-min audio: ~90s / ~3min
      - 1-hour audio: ~3min / ~6min
      - 3-hour meeting: ~10min / ~18min (plus 3-5min diarization)
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"error": f"unknown job_id: {job_id}"}
    return job.public()


@mcp.tool
def get_segments(job_id: str) -> dict[str, Any]:
    """Return per-segment timestamps and text for a completed job.

    Used for the dual-run merge workflow: call this on both the whisper-only
    job and the diarized job, then align by timestamp to get the best text
    per speaker turn.

    Each segment contains:
      - start: float (seconds from audio start)
      - end: float (seconds)
      - text: transcribed text for this segment
      - speaker: speaker ID (only present when the job used diarize=True)
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return {"error": f"unknown job_id: {job_id}"}
    if job.status != "done":
        return {"error": f"job not done yet, status={job.status}"}
    segs = job.raw_segments or []
    simplified = [
        {
            "start": round(s.get("start", 0.0), 2),
            "end": round(s.get("end", 0.0), 2),
            "text": s.get("text", "").strip(),
            **( {"speaker": s["speaker"]} if "speaker" in s else {} ),
        }
        for s in segs
    ]
    return {
        "job_id": job_id,
        "segment_count": len(simplified),
        "diarized": job.diarize and bool(job.speakers),
        "segments": simplified,
    }


@mcp.tool
async def transcribe(
    path: str,
    language: str | None = None,
    task: str = "transcribe",
    diarize: bool = False,
    num_speakers: int | None = None,
    wait_s: float = 45.0,
) -> dict[str, Any]:
    """Start a job and wait up to wait_s for it to complete inline.

    Returns the full transcript if done in time, otherwise returns job_id + status
    for continued polling via get_transcribe. Use start_transcribe directly for
    meetings or any audio longer than ~10 minutes to avoid client timeout issues.

    Args:
        path: Absolute path to a local audio file.
        language: Language code (e.g. "ru", "de"); omit to auto-detect.
        task: "transcribe" or "translate".
        diarize: Enable speaker diarization (requires HF_TOKEN).
        num_speakers: Known number of speakers.
        wait_s: Max seconds to wait inline (keep below client MCP timeout; 45 for 60s clients).
    """
    started = start_transcribe(
        path=path, language=language, task=task,
        diarize=diarize, num_speakers=num_speakers,
    )
    if "error" in started:
        return started

    job_id = started["job_id"]
    deadline = time.monotonic() + max(1.0, min(wait_s, 55.0))
    while time.monotonic() < deadline:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            return {"error": "job vanished"}
        if job.status in {"done", "error"}:
            return job.public()
        await asyncio.sleep(1.0)

    with _JOBS_LOCK:
        job = _JOBS[job_id]
    out = job.public()
    out["hint"] = (
        "Still running — call get_transcribe(job_id) every 30-60s. "
        "Typical: 30-min audio ≈ 90s warm; 3h audio ≈ 10-18min."
    )
    return out


@mcp.tool
def warm_model() -> dict[str, Any]:
    """Pre-load the Whisper model into Metal memory to avoid cold-start latency.

    Call this before transcribe when startup time matters (~5-15s cold).
    Returns once the model is resident in Metal memory.
    """
    global _model_loaded, _last_used
    with _model_lock:
        try:
            import mlx_whisper.load_models as lm
            lm.load_model(MODEL_REPO)
            _model_loaded = True
            _last_used = time.monotonic()
            LOG.info("model pre-loaded: %s", MODEL_REPO)
            return {"status": "loaded", "model": MODEL_REPO}
        except Exception as exc:
            LOG.error("warm_model failed: %s", exc)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool
def status() -> dict[str, Any]:
    """Report daemon health and all active/recent jobs."""
    with _JOBS_LOCK:
        jobs = [j.public() for j in _JOBS.values()]
    return {
        "model_loaded": _model_loaded,
        "model_repo": MODEL_REPO,
        "idle_unload_seconds": IDLE_UNLOAD_S,
        "seconds_since_last_use": round(time.monotonic() - _last_used, 1) if _last_used else None,
        "chunk_duration_s": CHUNK_DURATION_S,
        "diarization_available": bool(os.environ.get("HF_TOKEN", "").strip()),
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "jobs": jobs,
    }


# ---- entry point -------------------------------------------------------------

def main() -> None:
    host = os.environ.get("WHISPER_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("WHISPER_MCP_PORT", "8765"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(f"refusing to bind non-loopback host: {host}")
    LOG.info(
        "starting whisper-mcp on http://%s:%d/mcp — model: %s",
        host, port, MODEL_REPO,
    )
    mcp.run(transport="http", host=host, port=port, stateless_http=True)


if __name__ == "__main__":
    main()
