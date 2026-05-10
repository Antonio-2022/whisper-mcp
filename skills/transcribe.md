---
name: transcribe
description: High-quality audio transcription using the local whisper-mcp server. Use when transcribing meetings, interviews, calls, or any audio file. Handles language detection, speaker diarization for multi-speaker recordings, and large files over 3 hours automatically.
---

# Audio Transcription Skill

Produces near-original-quality transcripts from audio files using the local
whisper-mcp MCP server (mlx-whisper large-v3-turbo on Apple Silicon, MPS-accelerated
speaker diarization via pyannote.audio).

## Step 1: Gather required information

Determine or ask the user for:

- **File path** — must be absolute (e.g. `/Users/antonio/Downloads/meeting.m4a`). Ask if not provided.
- **Language** — specify explicitly whenever you know it:
  - Russian → `"ru"` | German → `"de"` | English → `"en"` | Spanish → `"es"` | French → `"fr"`
  - **This is the #1 quality lever.** Wrong or missing language produces garbled output.
  - Use `null` only when you genuinely cannot tell — the server auto-detects via a 30-second sample.
- **Multiple speakers?** — set `diarize: true` for any meeting, interview, panel, or call with 2+ people.
- **Speaker count** — if known (e.g. "3 people in this call"), pass `num_speakers` for better accuracy.
- **Task** — `"transcribe"` (default, same language) or `"translate"` (output in English regardless of input).

## Step 2: Warm the model (first transcription of a session)

If this is the first transcription and speed matters, pre-load the model to avoid 5-15s cold-start:

```
warm_model()
```

Skip this if the user just wants to queue without waiting for warm-up confirmation.

## Step 3: Start the transcription job

Always use `start_transcribe` — never use `transcribe` for files longer than ~5 minutes,
because MCP client timeouts will cut off the call before it finishes.

```
start_transcribe(
  path="/absolute/path/to/audio.m4a",
  language="ru",          ← always set if known; null only as last resort
  task="transcribe",
  diarize=true,           ← true for meetings / calls / interviews
  num_speakers=3          ← optional but improves accuracy
)
```

**Critical notes:**
- `language` specification is the single most important quality parameter. A missed language
  detection produces phonetically plausible but semantically wrong text in the wrong script.
- Files >30 min are automatically split into 30-minute chunks and merged — no manual action needed.
- Diarization runs after transcription as a separate phase (~2-5 min for a 1-hour recording).
- The tool returns immediately with a `job_id` and `status: "queued"`.

## Step 4: Poll until complete

After starting, inform the user ("Transcription job started, polling for results…"), then poll:

```
get_transcribe(job_id="<job_id from step 3>")
```

**Typical completion times:**

| Audio length | Warm model | Cold model |
|---|---|---|
| 10 min | ~30s | ~60s |
| 30 min | ~90s | ~3 min |
| 1 hour | ~3 min | ~6 min |
| 3 hours | ~10 min | ~18 min |
| + diarization | +2-5 min | +2-5 min |

Poll every 30-60 seconds. Report elapsed time to the user between polls.
Stop when `status` is `"done"` or `"error"`.

## Step 5: Process and present the result

When `status == "done"`, the response contains:

| Field | Content |
|---|---|
| `text` | Full plain transcript (no speaker labels) |
| `transcript_with_speakers` | Speaker-labeled blocks — use this for meetings |
| `speakers` | Speaker IDs e.g. `["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]` |
| `language` | Language code that was used/detected |
| `audio_duration_s` | Total audio length in seconds |
| `segments` | Number of transcription segments |

**Always prefer `transcript_with_speakers`** over `text` when it is present.

### Output size handling

- **< 20,000 characters** — display inline with header (language, duration, speakers)
- **≥ 20,000 characters** — write to a file:

```
/Users/antonio/Downloads/transcript-YYYY-MM-DD.txt
```

Then show: language detected, total duration, speaker list, and first 500 characters as preview.

### Speaker name substitution

After diarization, ask: *"Do you want to rename the speakers? (e.g. SPEAKER_00 → Antonio)"*

If yes, do a global find-replace on the transcript. Example:
- `[SPEAKER_00]` → `[Antonio]`
- `[SPEAKER_01]` → `[Maria]`

## Step 6: Quality check before presenting

Before reporting the transcript as done:
- [ ] `language` field matches the expected audio language
- [ ] If `diarize=true` was requested, `transcript_with_speakers` is present
- [ ] Text is non-empty and not phonetically garbled (garbled = wrong language used)
- [ ] For large outputs, file was written successfully and path confirmed

**If quality looks wrong:** re-run with the correct explicit language code.

---

## Error handling

| Error | Action |
|---|---|
| `"file not found"` | Confirm the absolute path with the user |
| `"HF_TOKEN not set"` | Diarization unavailable; retry with `diarize=false` |
| `error` contains "diarization failed; plain transcript still available" | Use `text` field; note diarization failed |
| Job stays `"running"` > 30 min | Call `status()` to check daemon health |
| Text looks garbled / wrong script | Re-run with explicit `language` parameter |

---

## Quick reference — common invocations

**Russian meeting, 3 speakers:**
```
start_transcribe(path="/path/to/meeting.m4a", language="ru", diarize=true, num_speakers=3)
```

**English interview, unknown speakers:**
```
start_transcribe(path="/path/to/interview.mp3", language="en", diarize=true)
```

**German lecture, single speaker, translate to English:**
```
start_transcribe(path="/path/to/lecture.wav", language="de", task="translate", diarize=false)
```

**Unknown language (auto-detect):**
```
start_transcribe(path="/path/to/audio.m4a", language=null, diarize=false)
```
