# Development Guide for whisper-subtitler

This guide provides an overview of the whisper-subtitler architecture and module structure for developers.

## Architectural Overview

whisper-subtitler uses a modular component-based architecture that separates concerns and promotes maintainability:

```
whisper-subtitler/
├── modules/
│   ├── audio/          # Audio extraction from video files
│   ├── config.py       # Configuration management
│   ├── diarisation/    # Speaker identification
│   ├── logger.py       # Logging utilities
│   ├── output/         # Output format generators
│   ├── transcribe/     # Whisper transcription
│   └── cli.py          # Command-line interface
└── main.py             # Application entry point
```

## Core Components

### 1. Configuration (config.py)

Manages configuration from multiple sources:
- Environment variables (.env files)
- Command-line arguments
- Configuration files

### 2. Audio Extraction (audio/extractor.py)

Extracts audio from video files using FFmpeg with configurable settings for sample rate and channels.

### 3. Transcription (transcribe/transcriber.py)

Handles speech-to-text conversion using Whisper with configurable model sizes and language settings.

### 4. Speaker Diarization (diarisation/diarizer.py)

Identifies different speakers in the audio using Pyannote.audio with support for:
- Known speaker count
- Speaker clustering
- CUDA acceleration

### 5. Output Formatting (output/*.py)

Converts transcription results into various subtitle formats:
- Plain text (TXT)
- SubRip Subtitle (SRT)
- WebVTT (VTT)
- Timed Text Markup Language (TTML)

### 6. Logging (logger.py)

Provides configurable logging capabilities with support for file output and different log levels.

### 7. Command Line Interface (cli.py)

Provides a user-friendly interface using argparse with extensive command-line options.

## Application Flow

1. Parse command-line arguments
2. Load configuration from .env and config files
3. Extract audio from video file
4. Transcribe audio using Whisper
5. Identify speakers using Pyannote (if enabled)
6. Combine transcription and speaker information
7. Generate output files in requested formats

## Adding New Features

### Adding a New Output Format

1. Create a new formatter class in `modules/output/formats.py`
2. Register the formatter in `modules/output/formatter.py`
3. Add the format to the CLI options in `modules/cli.py`

### Adding Configuration Options

1. Add default value in the `Config` class in `modules/config.py`
2. Add environment variable loading in `load_from_env`
3. Add CLI argument in `modules/cli.py`

## Testing

Run the application with various settings to ensure it works as expected:

```bash
# Basic test
python run.py transcribe .local/test1.mp4 --force

# Test with no diarization
python run.py transcribe .local/test1.mp4 --no-diarization --force

# Test with custom model
python run.py transcribe .local/test1.mp4 -m small --force
``` 