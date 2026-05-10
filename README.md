# whisper-mcp

Local-only MCP server for high-quality audio transcription on Apple Silicon (MLX).

## Features

- Fast transcription via **mlx-whisper** (`whisper-large-v3-turbo`, Metal/MLX)
- **Automatic language detection** — 30-second sample pass when language not specified
- **Chunked transcription** for recordings > 30 minutes (splits into 30-min WAV chunks, merges results)
- **Speaker diarization** via `pyannote.audio` 3.x on MPS (requires HuggingFace token)
- Background job queue with polling — survives MCP client timeouts

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.11+ (via pyenv)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### HuggingFace token (for speaker diarization)

1. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Create a read token at https://huggingface.co/settings/tokens
3. Store locally — **never commit this file**:

```bash
mkdir -p ~/.config/whisper-mcp
echo 'HF_TOKEN=hf_YOUR_TOKEN_HERE' > ~/.config/whisper-mcp/.env
chmod 600 ~/.config/whisper-mcp/.env
```

### launchd service (macOS)

```bash
cp launchd/com.antonio.whisper-mcp.plist.template ~/Library/LaunchAgents/com.antonio.whisper-mcp.plist
# Edit the plist and replace ${HF_TOKEN} with your actual token
# Optionally adjust the Python path to match your pyenv version
launchctl load ~/Library/LaunchAgents/com.antonio.whisper-mcp.plist
```

### Claude Code MCP configuration

Add to your `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "whisper": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

## MCP Tools

| Tool | Description |
|---|---|
| `start_transcribe` | Start a background job — returns `job_id` immediately |
| `get_transcribe` | Poll a job for status and result |
| `transcribe` | Start + wait up to N seconds inline (use for short files) |
| `warm_model` | Pre-load model to avoid cold-start latency (~5-15s) |
| `status` | Daemon health, active jobs, and configuration |

### Key parameters for `start_transcribe`

| Parameter | Type | Description |
|---|---|---|
| `path` | string | Absolute path to local audio file (m4a, mp3, wav, mp4, opus…) |
| `language` | string \| null | BCP-47 code e.g. `"ru"`, `"de"`, `"en"`. Omit to auto-detect. **Always set if known.** |
| `task` | string | `"transcribe"` (default) or `"translate"` (→ English) |
| `diarize` | bool | `true` to identify speakers. Requires HF_TOKEN. |
| `num_speakers` | int \| null | Known speaker count — improves diarization accuracy |

### Result fields (when `status == "done"`)

| Field | Description |
|---|---|
| `text` | Plain transcript, no speaker labels |
| `transcript_with_speakers` | Speaker-labeled blocks (present when `diarize=true`) |
| `speakers` | List of speaker IDs e.g. `["SPEAKER_00", "SPEAKER_01"]` |
| `language` | Detected/used language code |
| `audio_duration_s` | Total audio duration in seconds |

## Typical completion times

| Audio length | Warm model | Cold model |
|---|---|---|
| 10 min | ~30s | ~60s |
| 30 min | ~90s | ~3min |
| 1 hour | ~3min | ~6min |
| 3 hours | ~10min | ~18min (+3-5min diarization) |

## Logs

```bash
tail -f ~/Library/Logs/whisper-mcp.log
```

Rotating log, max 10 MB × 3 files.
