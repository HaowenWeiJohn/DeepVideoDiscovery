# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Video Discovery (DVD) is an agentic video understanding system that uses LLMs with tool-calling to answer questions about long-form videos (hour+ duration). It achieves state-of-the-art performance on benchmarks like LVBench by treating video clips as a searchable database and using an autonomous observe-reason-act loop.

**Paper:** "Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding" (NeurIPS 2025)

## Commands

### Run on a YouTube video
```bash
python local_run.py "https://www.youtube.com/watch?v=VIDEO_ID" "your question here"
```

### Launch Gradio demo
```bash
python app.py --share  # --share for public URL
```

### Reproduce LVBench benchmark
```bash
# Set database path
export DATABASE_DIR=/path/to/database

# Prepare database from pre-built captions
python -m reproduce.prepare_database /path/to/zipfile $DATABASE_DIR

# Decode video frames
python -m reproduce.decode_frames --part $DATABASE_DIR/LVBench_4.1/all_videos_split.zip.001 --out $DATABASE_DIR/LVBench_4.1 --fps 2

# Run benchmark
python -m reproduce.run_benchmark $DATABASE_DIR/LVBench_4.1 $DATABASE_DIR/LVBench_4.1/video_info.meta.jsonl
```

### Install dependencies
```bash
pip install -r requirements.txt
pip install gradio  # optional, for demo
```

## Architecture

### Two-Stage Pipeline

**Stage 1: Multi-granular Video Database Construction** (`dvd/build_database.py`, `dvd/frame_caption.py`)
- Video is segmented into clips (default: 10 seconds, configurable via `CLIP_SECS` in `config.py`)
- Frames decoded at 2 FPS (configurable via `VIDEO_FPS`)
- Each clip is captioned by VLM (GPT-4.1-mini by default)
- Captions are embedded using `text-embedding-3-large` and stored in `NanoVectorDB`
- Subject registry tracks characters/objects across clips
- Database stored as JSON with embeddings for semantic search

**Stage 2: Agentic Search and Answer (ASA)** (`dvd/dvd_core.py`)
- `DVDCoreAgent` implements ReAct-style observe-reason-act loop
- Uses OpenAI function calling with reasoning model (o3 by default)
- Maximum iterations configurable via `MAX_ITERATIONS` in config

### Search Tools (defined in `dvd/build_database.py`)

| Tool | Purpose | Returns |
|------|---------|---------|
| `global_browse_tool` | Get video overview, subject registry, and query-relevant events | Subject registry + event summary |
| `clip_search_tool` | Semantic search over clip captions | Top-k ranked clips with timestamps |
| `frame_inspect_tool` | Fine-grained VQA on specific time ranges | VLM response about frame details |

### Operating Modes

- **Full Mode** (`LITE_MODE=False`): Downloads video, extracts frames, generates VLM captions. All three search tools available.
- **Lite Mode** (`LITE_MODE=True`): Uses only SRT subtitles (no video download/VLM captioning). `frame_inspect_tool` is disabled. Good for YouTube podcasts.

## Key Files

| File | Purpose |
|------|---------|
| `dvd/config.py` | All configuration: API endpoints, model names, clip settings |
| `dvd/dvd_core.py` | Main agent class `DVDCoreAgent` with ReAct loop |
| `dvd/build_database.py` | Tool implementations and database initialization |
| `dvd/frame_caption.py` | Video captioning pipeline with VLM |
| `dvd/utils.py` | OpenAI/Azure API calls, embedding service |
| `local_run.py` | CLI entry point for single video Q&A |
| `app.py` | Gradio web demo |

## Configuration

Set in `dvd/config.py`:

```python
# API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set this for OpenAI API
# Or configure AOAI_*_ENDPOINT_LIST for Azure OpenAI

# Model Selection
AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"      # Database captioning
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"           # Agent reasoning
AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"         # Frame inspect tool

# Agent Settings
LITE_MODE = True                # Use only subtitles (no video download)
MAX_ITERATIONS = 3              # Max reasoning steps
CLIP_SECS = 10                  # Clip duration for segmentation
VIDEO_FPS = 2                   # Frame extraction rate
SINGLE_CHOICE_QA = True         # For benchmarks: agent returns option letters
```

## Data Flow

1. **Input:** YouTube URL or video path + question
2. **Video Processing:** Download → Frame extraction → VLM captioning → Embedding
3. **Database:** `video_database/{video_id}/` contains:
   - `frames/` - Extracted video frames (if full mode)
   - `captions/captions.json` - Clip captions and subject registry
   - `database.json` - NanoVectorDB with embeddings
4. **Agent Loop:** Query → Tool selection → Observation → Reasoning → Answer
5. **Output:** Final answer extracted via `finish()` tool call

## Agent Behavior Patterns

The agent autonomously chooses search strategies:
- **Simple Action:** Global browse → Clip search → Frame inspect → Answer
- **Iterative Search:** Multiple clip search/frame inspect rounds for complex queries
- **Global Browse Only:** Sufficient for high-level questions about video type/content

### Agent Methods

- `run(question)`: Single synchronous execution, returns message history
- `stream_run(question)`: Generator that yields messages as they're produced (for UI streaming)
- `parallel_run(questions, max_workers=4)`: Run multiple questions concurrently on the same video
