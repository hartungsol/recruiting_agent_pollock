# Recruiting Agent Pollock

A multi-agent recruiting interview system that integrates with a Pollock/OSCAR-style defeasible reasoner, PostgreSQL, and an LLM backend.

## Architecture Overview

This system implements a multi-agent architecture for conducting and evaluating recruiting interviews:

- **Orchestrator**: Coordinates interview flow and manages state across agents
- **Agents**: Specialized components for intent classification, question planning, parsing, QA, red flag detection, and scoring
- **Reasoning**: Integration with OSCAR-style defeasible reasoning for nuanced decision-making
- **Retrieval**: Vector store integration for semantic search over job descriptions, resumes, and interview knowledge
- **IO**: Text and voice interfaces for interview interaction
- **DB**: PostgreSQL-backed persistence layer

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and configure your environment variables:

```bash
cp .env.example .env
```

## Usage

```bash
# Run the main application
python -m recruiting_agent_pollock.main

# Run in voice mode (optional)
python -m recruiting_agent_pollock.main --mode voice

# Non-interactive voice runner (loads job + candidate from args)
python scripts/voice_interview.py --candidate-name "Ted" --stdin-job

# Voice runner with optional human interviewer turns
python scripts/voice_interview.py --candidate-name "Ted" --stdin-job --human-interviewer

# Run tests
pytest
```

## Voice Mode (Optional)

Install optional Python deps:

```bash
pip install -e ".[dev,voice]"
```

System deps:
- Linux: `sounddevice` requires PortAudio (Debian/Ubuntu: `sudo apt-get install portaudio19-dev`)

Offline STT/TTS:
- STT: install `faster-whisper` (Python). Default is CPU; optionally set `POLL0CK_STT_MODEL` and `POLL0CK_STT_DEVICE=cpu|cuda`
- TTS (AI interviewer speech): install `piper` (CLI) and download a Piper `.onnx` model, then set `POLL0CK_PIPER_MODEL=/path/to/model.onnx` (if missing, voice mode falls back to printing responses)

Note (Linux): some distros ship a different `/usr/bin/piper` (a GTK app) that is NOT Piper TTS. If you see errors like `Unknown option --model`, install Piper TTS and point to it with `POLL0CK_PIPER_BIN`.

Example (Linux/Mac):
```bash
export POLL0CK_PIPER_BIN=/absolute/path/to/piper
export POLL0CK_PIPER_MODEL=/absolute/path/to/voice.onnx
python -m recruiting_agent_pollock.main --mode voice
```

Quick validation:
```bash
"$POLL0CK_PIPER_BIN" --help | grep -E -- '--model|--output_file'
```

Artifacts:
- Voice sessions write artifacts under `data/interviews/{interview_id}/` (WAVs, transcripts, and `turns.jsonl`).

Human interviewer turns (optional):
- In `scripts/voice_interview.py`, pass `--human-interviewer` and choose `i` at the prompt to record/transcribe a human interviewer utterance. This is recorded into the orchestrator transcript/memory so the next candidate turn has the right context.

## Project Structure

```
recruiting_agent_pollock/
├── src/
│   └── recruiting_agent_pollock/
│       ├── orchestrator/     # Interview orchestration and state management
│       ├── agents/           # Specialized interview agents
│       ├── reasoning/        # OSCAR/Pollock defeasible reasoner integration
│       ├── retrieval/        # Vector store and semantic search
│       ├── io/               # Text and voice interfaces
│       ├── db/               # Database models and repository
│       └── models/           # LLM client abstraction
└── tests/                    # Test suite
```

## License

MIT
