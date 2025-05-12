# whisper-subtitler Documentation

Welcome to the official documentation for whisper-subtitler, an automatic subtitle generator with speaker identification.

## Documentation Index

### User Documentation

- [Installation Guide](installation-guide.md) - How to install and set up whisper-subtitler
- [User Guide](user-guide.md) - Comprehensive guide to using whisper-subtitler
- [Usage Examples](usage-examples.md) - Common usage patterns and examples
- [FAQ](faq.md) - Frequently asked questions and troubleshooting

### Developer Documentation

- [Development Guide](development.md) - Overview of the architecture and development workflow
- [API Reference](api-reference.md) - Detailed reference of classes, methods, and data structures

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/picommcapp/whisper-subtitler.git
cd whisper-subtitler

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.sample .env
# Then edit .env with your HuggingFace token
```

### Basic Usage

```bash
# Transcribe a video with speaker identification
python run.py transcribe video.mp4

# Transcribe without speaker identification
python run.py transcribe video.mp4 --no-diarization

# Use a specific Whisper model
python run.py transcribe video.mp4 -m medium

# Specify output directory
python run.py transcribe video.mp4 -o output_dir
```

See the [User Guide](user-guide.md) for detailed instructions.

## Features

- **High-quality transcription** using OpenAI's Whisper
- **Speaker identification** using Pyannote.audio
- **Multiple output formats**: TXT, SRT, VTT, TTML
- **GPU acceleration** for faster processing
- **Configurable** through command-line options or environment variables
- **Advanced diarization options** for improved speaker identification
- **Support for 100+ languages**

## Contributing

Contributions are welcome! See the [Development Guide](development.md) for information on how to contribute to the project.

## License

whisper-subtitler is released under the [MIT License](https://github.com/picommcapp/whisper-subtitler/blob/main/LICENSE).
