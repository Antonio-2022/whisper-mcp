---
name: transcribe
description: High-quality audio transcription using the local whisper-mcp server. Use when transcribing meetings, interviews, calls, or any audio file. Implements a dual-run merge workflow: whisper-only for best text quality + whisper+diarization for speaker labels, then merges by timestamp to produce a speaker-labeled dialogue with the best phrasing per turn.
---

# Audio Transcription Skill

Produces speaker-labeled transcripts at near-original quality by running two parallel
jobs and merging them: a clean whisper-only pass for text fidelity, a diarized pass for
speaker attribution, then a timestamp-aligned merge picking the best phrasing per turn.

---

## Step 1: Gather required information

Ask the user or infer from context:

- **File path** — absolute path to a local audio file. Ask if not provided.
- **Language** — the single most important quality parameter:
  - Russian → `"ru"` | German → `"de"` | English → `"en"` | Spanish → `"es"` | French → `"fr"`
  - Wrong or missing language = garbled output. Use `null` only when genuinely unknown.
- **Number of speakers** — if known (e.g. "3 founders in this call"), pass `num_speakers`. Improves diarization.
- **Task** — `"transcribe"` (default) or `"translate"` (→ English).

---

## Step 2: Start both jobs simultaneously

Fire both jobs **at the same time** — they queue independently in the background.

**Job A — whisper-only (best text quality):**
```
start_transcribe(
  path="/absolute/path/to/audio.m4a",
  language="ru",
  task="transcribe",
  diarize=false
)
→ save as job_a_id
```

**Job B — whisper + diarization (speaker labels):**
```
start_transcribe(
  path="/absolute/path/to/audio.m4a",
  language="ru",
  task="transcribe",
  diarize=true,
  num_speakers=3        ← set if known
)
→ save as job_b_id
```

Both transcription runs use the same language setting and the same mlx-whisper model.
Job B additionally runs pyannote speaker diarization after transcription completes.

---

## Step 3: Poll both jobs until done

Poll every 30-60 seconds. Report progress to the user.

```
get_transcribe(job_id=job_a_id)   → check status
get_transcribe(job_id=job_b_id)   → check status
```

**Typical completion times:**

| Audio | Job A (whisper only) | Job B (+ diarization) |
|---|---|---|
| 10 min | ~30s | ~3-5 min |
| 30 min | ~90s | ~5-8 min |
| 1 hour | ~3 min | ~8-12 min |
| 3 hours | ~10-18 min | ~20-30 min |

Job A finishes well before Job B. Once Job A is done, retrieve its segments immediately
(Step 4) to have them ready for the merge.

---

## Step 4: Retrieve segments from both jobs

Once **both** jobs report `status: "done"`:

```
get_segments(job_id=job_a_id)   → run1_segments  (fields: start, end, text)
get_segments(job_id=job_b_id)   → run2_segments  (fields: start, end, text, speaker)
```

---

## Step 5: Merge — timestamp-aligned, best-text-wins

For each **speaker turn** in Job B, extract the corresponding text from Job A.

**Algorithm (implement in-context as Python):**

```python
def merge_runs(run1_segs, run2_segs):
    """
    run1_segs: [{start, end, text}]              — best text quality, no speakers
    run2_segs: [{start, end, text, speaker}]     — speaker labels, slightly lower quality text

    Strategy:
      1. Group consecutive same-speaker segments in run2 into "turns"
      2. For each turn, collect all run1 segments that overlap its time span
      3. Use run1 text as the canonical content for that turn (better phrasing)
      4. Fall back to run2 text if no run1 overlap found
    """
    # Step 1: group run2 into speaker turns
    turns = []
    for seg in run2_segs:
        sp = seg.get("speaker", "UNKNOWN")
        if turns and turns[-1]["speaker"] == sp:
            turns[-1]["end"] = seg["end"]
        else:
            turns.append({"speaker": sp, "start": seg["start"], "end": seg["end"]})

    # Step 2-4: fill each turn with run1 text
    result = []
    for turn in turns:
        t_start, t_end = turn["start"], turn["end"]

        # Collect overlapping run1 segments (overlap = run1 ends after turn starts AND starts before turn ends)
        r1_overlap = [s for s in run1_segs if s["end"] > t_start and s["start"] < t_end]
        if r1_overlap:
            text = " ".join(s["text"].strip() for s in r1_overlap if s["text"].strip())
        else:
            # Fallback: use run2 text for this turn
            r2_overlap = [s for s in run2_segs
                          if s.get("speaker") == turn["speaker"]
                          and s["end"] > t_start and s["start"] < t_end]
            text = " ".join(s["text"].strip() for s in r2_overlap if s["text"].strip())

        if text:
            result.append({
                "speaker": turn["speaker"],
                "start": round(t_start, 1),
                "end":   round(t_end, 1),
                "text":  text,
            })

    return result
```

Run this merge on the two segment lists received from `get_segments`.

---

## Step 6: Format as speaker-labeled dialogue

```python
def format_dialogue(merged_turns, speaker_names=None):
    """
    merged_turns: [{speaker, start, end, text}]
    speaker_names: optional dict, e.g. {"SPEAKER_00": "Antonio", "SPEAKER_01": "Maria"}
    """
    lines = []
    for turn in merged_turns:
        sp = turn["speaker"]
        label = speaker_names.get(sp, sp) if speaker_names else sp
        lines.append(f"[{label}]")
        lines.append(turn["text"])
        lines.append("")
    return "\n".join(lines).strip()
```

**Ask the user if they want to rename speakers** before formatting:
- e.g. `SPEAKER_00 → Antonio`, `SPEAKER_01 → Maria`

If yes, substitute before calling `format_dialogue`.

---

## Step 7: Present and save the result

- **< 20,000 chars** — display inline with header (language, duration, speakers)
- **≥ 20,000 chars** — write to file: `/Users/antonio/Downloads/transcript-YYYY-MM-DD.txt`

Show summary: language detected, audio duration, speaker list, turn count, first 500 chars.

---

## Quality checklist before reporting done

- [ ] `language` field matches expected audio language (if garbled, re-run with explicit code)
- [ ] `transcript_with_speakers` / merged dialogue contains `[SPEAKER_XX]` or named labels
- [ ] Both jobs completed with `status: "done"` before merge
- [ ] Merged turn count is non-zero
- [ ] Large output was written to file successfully

---

## Error handling

| Error | Action |
|---|---|
| Job A or B shows `status: "error"` | Check `error` field; most common: wrong path or language |
| `"diarization failed; plain transcript still available"` | Job B's text is still usable for run2_segs; speaker labels may be missing |
| `"HF_TOKEN not set"` | Diarize unavailable — use Job A text only, no speaker labels |
| Garbled text in run1_segs | Re-run Job A with explicit `language=` code |
| Merge produces empty turns | Check segment timestamps overlap — may need to widen overlap threshold |
| Job stays `"running"` > 30 min | Call `status()` to check daemon health |

---

## Quick-start examples

**Russian founders call, 3 speakers:**
```
Job A: start_transcribe(path="...", language="ru", diarize=false)
Job B: start_transcribe(path="...", language="ru", diarize=true, num_speakers=3)
→ merge → rename SPEAKER_00/01/02 → dialogue
```

**English interview, unknown speaker count:**
```
Job A: start_transcribe(path="...", language="en", diarize=false)
Job B: start_transcribe(path="...", language="en", diarize=true)
→ merge → dialogue
```

**Single-speaker lecture (no diarization needed):**
```
start_transcribe(path="...", language="de", diarize=false)
→ plain text, no merge step needed
```
