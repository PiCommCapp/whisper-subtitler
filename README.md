# whisper-subtitler

[![Release](https://img.shields.io/github/v/release/picommcapp/whisper-subtitler)](https://img.shields.io/github/v/release/picommcapp/whisper-subtitler)
[![Build status](https://img.shields.io/github/actions/workflow/status/picommcapp/whisper-subtitler/main.yml?branch=main)](https://github.com/picommcapp/whisper-subtitler/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/picommcapp/whisper-subtitler/branch/main/graph/badge.svg)](https://codecov.io/gh/picommcapp/whisper-subtitler)
[![Commit activity](https://img.shields.io/github/commit-activity/m/picommcapp/whisper-subtitler)](https://img.shields.io/github/commit-activity/m/picommcapp/whisper-subtitler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic subtitle generator with speaker identification using OpenAI's Whisper and Pyannote.

- **Github repository**: <https://github.com/picommcapp/whisper-subtitler/>
- **Documentation** <https://picommcapp.github.io/whisper-subtitler/>

## Features

- Automatic transcription using OpenAI's Whisper
- Speaker diarization (identification) using Pyannote.audio
- Multiple output formats (TXT, SRT, VTT, TTML)
- Configurable via .env file or command line
- GPU acceleration support

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file based on `.env.sample`
4. Get a HuggingFace token from https://hf.co/settings/tokens
5. Add the token to your `.env` file

## Quick Usage

```bash
# Basic usage - outputs to same directory as input
python run.py transcribe path/to/video.mp4

# Skip speaker identification
python run.py transcribe path/to/video.mp4 --no-diarization

# Specify output directory
python run.py transcribe path/to/video.mp4 -o path/to/output

# Use specific model size
python run.py transcribe path/to/video.mp4 -m medium
```

See [Usage Examples](docs/usage-examples.md) for more examples.

## Configuration

The application can be configured in three ways:

1. **Command Line Arguments**: Direct options when running the command
2. **Environment Variables**: Set in a `.env` file
3. **Configuration File**: Advanced settings in a separate config file

### Environment Variables

Create a `.env` file in the project root with these options:

```bash
# Required for speaker identification
HUGGINGFACE_TOKEN=your_token_here

# Whisper model configuration 
WHISPER_MODEL_SIZE=medium   # tiny, base, small, medium, large
WHISPER_LANGUAGE=en         # language code or empty for auto-detection

# Speaker diarization settings
SKIP_DIARIZATION=false      # set to true to disable speaker identification
NUM_SPEAKERS=               # number of speakers if known

# Output settings
OUTPUT_FORMATS=txt,srt,vtt,ttml
```

For more options, see the `.env.sample` file.

## Development

See [Development Guide](docs/development.md) for information on development workflow.

### API Compatibility Notes

- **Pyannote.audio**: The speaker diarization module includes compatibility fixes for newer versions of Pyannote.audio. If the standard speaker clustering fails (which may happen with certain API versions), an alternative clustering method is automatically applied as a fallback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

