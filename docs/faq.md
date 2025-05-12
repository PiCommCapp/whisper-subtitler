# Frequently Asked Questions

## General Questions

### What is whisper-subtitler?

whisper-subtitler is an open-source tool that automatically generates subtitles from video files, including speaker identification. It uses OpenAI's Whisper for speech recognition and Pyannote.audio for speaker diarization.

### What file formats are supported?

For input, whisper-subtitler supports any video format that FFmpeg can process, including:
- MP4, MOV, AVI, MKV, WMV, FLV, etc.

For output, it generates subtitles in these formats:
- TXT (plain text transcript)
- SRT (SubRip subtitle format)
- VTT (WebVTT format for web videos)
- TTML (Timed Text Markup Language)

### What languages are supported?

whisper-subtitler supports the same languages as Whisper, which includes 100+ languages. Some of the well-supported languages include:
- English, Spanish, French, German, Italian, Portuguese, Dutch
- Chinese, Japanese, Korean, Russian, Arabic, Hindi
- And many more

For the full list, refer to [Whisper's documentation](https://github.com/openai/whisper).

### How accurate is the transcription?

Accuracy depends on several factors:
- The model size you choose (larger models are more accurate)
- Audio quality (clear audio with minimal background noise works best)
- Speaker clarity and accent
- Language (English tends to have the highest accuracy)

With the "large" model and clear audio, you can expect accuracy rates of 90%+ for English content.

### How accurate is speaker identification?

Speaker identification (diarization) is typically 80-90% accurate in ideal conditions. Factors affecting accuracy include:
- Number of speakers (more speakers = lower accuracy)
- Speaker overlap (when people talk over each other)
- Audio quality and background noise
- Speaker similarity (similar voices are harder to distinguish)

Using the `--speakers` option when you know the number of speakers can improve accuracy.

## Configuration and Usage

### Do I need a HuggingFace token?

Yes, but only if you want to use speaker identification (diarization). The token is required to download and use Pyannote.audio's speaker diarization models.

If you don't want or need speaker identification, use the `--no-diarization` flag and no token is required.

### Which Whisper model should I choose?

The choice depends on your priorities:

- **tiny/base**: Fastest, lowest accuracy, minimal resource usage. Good for quick drafts.
- **small**: Good balance of speed and accuracy for casual use.
- **medium**: Good accuracy with reasonable speed. Recommended for most cases.
- **large**: Highest accuracy, slowest, requires more RAM. Best for professional use.

### How can I make transcription faster?

To improve speed:
1. Use a smaller model (`-m small` or `-m tiny`)
2. Use GPU acceleration (enabled by default with CUDA)
3. Skip diarization with `--no-diarization`
4. Process shorter audio segments rather than long videos

### Can I run without a GPU?

Yes, whisper-subtitler works without a GPU, but it will be significantly slower, especially with larger models. For CPU-only mode, use the `--cpu` flag.

### How much disk space do the models require?

The Whisper models vary in size:
- tiny: 39MB
- base: 74MB
- small: 244MB
- medium: 769MB
- large: 1.5GB

The Pyannote diarization models add about 1GB more.

### Where are the model files stored?

By default, the models are downloaded to:
- Whisper models: `~/.cache/whisper/`
- Pyannote models: `~/.cache/torch/pyannote/`

## Troubleshooting

### Error: "HUGGINGFACE_TOKEN is required for speaker diarization"

This error occurs when trying to use speaker diarization without providing a HuggingFace token. You have two options:

1. Add your token to the `.env` file:
   ```
   HUGGINGFACE_TOKEN=hf_your_token_here
   ```

2. Pass the token via command line:
   ```bash
   python run.py transcribe video.mp4 --token hf_your_token_here
   ```

3. Skip diarization entirely:
   ```bash
   python run.py transcribe video.mp4 --no-diarization
   ```

### Error: "CUDA is not available"

This warning appears when trying to use GPU acceleration, but CUDA is not available on your system. Possible solutions:

1. Install CUDA if you have an NVIDIA GPU
2. Use CPU mode with the `--cpu` flag
3. Check that your NVIDIA drivers are up-to-date

### Error: "FFmpeg not found"

This error occurs when FFmpeg is not installed or not in your PATH:

1. Install FFmpeg for your operating system
2. Make sure it's added to your PATH environment variable
3. Verify with `ffmpeg -version` that it's accessible

### The transcription quality is poor

If you're getting low-quality transcriptions:

1. Try a larger model (`-m large`)
2. Check the audio quality of your source file
3. If the language is not English, specify the language with `-l [language_code]`
4. Consider preprocessing the audio to improve quality (noise reduction, normalization)

### Speaker labels are inconsistent

If speaker labels are inconsistent (same person labeled as different speakers):

1. Use the `--cluster` flag to enable speaker clustering
2. If you know the number of speakers, use `--speakers [count]`
3. Try the `--optimize-speakers` flag

### Out of memory error

If you encounter memory errors:

1. Use a smaller model (`-m small` or `-m tiny`)
2. Free up memory by closing other applications
3. For GPU memory issues, try using CPU mode with `--cpu`
4. Split your video into smaller chunks and process them separately

## Advanced Usage

### Can I batch process multiple files?

Yes, you can write a simple shell script:

```bash
# Bash example
for video in *.mp4; do
  python run.py transcribe "$video" --no-diarization
done
```

### Can I customize the output file names?

Currently, the output files use the same base name as the input file with different extensions. To change this, you'd need to modify the code or create a post-processing script.

### Can I integrate this with other tools?

Yes, whisper-subtitler can be integrated with other tools:

1. As a command-line tool in pipelines or scripts
2. As a Python library in your own code (see API Reference)
3. As part of video editing workflows by generating subtitle files

### Is there a web interface available?

The main whisper-subtitler project doesn't include a web interface, but you could:

1. Create a web UI using frameworks like Flask or FastAPI
2. Use the API to integrate with existing web applications
3. Look for community-contributed web interfaces

## Project and Support

### How can I contribute to the project?

Contributions are welcome! You can:

1. Submit bug reports or feature requests on GitHub
2. Contribute code through pull requests
3. Improve documentation
4. Share your use cases and examples

### Is there commercial support available?

whisper-subtitler is an open-source project without official commercial support. However, you may find community members or companies offering support services.

### How is this different from just using Whisper directly?

whisper-subtitler adds several features on top of Whisper:

1. Speaker identification using Pyannote.audio
2. Multiple subtitle format outputs
3. Simplified configuration and usage
4. Optimized processing pipeline
5. Better default settings for subtitle generation 