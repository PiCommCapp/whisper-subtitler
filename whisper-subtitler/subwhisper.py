import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import torch
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from tqdm import tqdm

# === LOGGING SETUP ===
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.error(
        "HUGGINGFACE_TOKEN environment variable not set. Please set it with your token from https://hf.co/settings/tokens"
    )
    logger.error("Also make sure to accept the user conditions at https://hf.co/pyannote/speaker-diarization")
    exit(1)

SHOW_SPEAKER_DEBUG = os.getenv("SHOW_SPEAKER_DEBUG", "False").lower() in ("1", "true", "yes")
TTML_TITLE = os.getenv("TTML_TITLE", "Transcription")
TTML_LANGUAGE = os.getenv("TTML_LANGUAGE", "en-GB")

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = Path(os.getenv("INPUT_FILE", str(SCRIPT_DIR / "../.local/test.mp4")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(SCRIPT_DIR / "../.local")))
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")  # tiny, base, small, medium, large

# === PATH RESOLUTION & SAFETY ===
input_path = INPUT_FILE.resolve()
output_path = OUTPUT_DIR.resolve()

if not input_path.exists():
    logger.error(f"Input file not found: {input_path}")
    exit(1)

output_path.mkdir(parents=True, exist_ok=True)
base_name = input_path.stem  # Filename without extension

# Extract audio from video
audio_path = output_path / f"{base_name}.wav"
logger.info(f"Extracting audio to: {audio_path}")
try:
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            os.getenv("AUDIO_SAMPLE_RATE", "16000"),
            "-ac",
            os.getenv("AUDIO_CHANNELS", "1"),
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
except subprocess.CalledProcessError as e:
    logger.error(f"Failed to extract audio: {e.stderr.decode()}")
    exit(1)

# === DIARISATION ===
logger.info("Running speaker diarisation...")

# Configure CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Configure Pyannote pipeline with performance settings
diag_auth = {"use_auth_token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    **diag_auth,
)

# Move pipeline to GPU if available
if device.type == "cuda":
    diar_pipeline = diar_pipeline.to(device)
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

diarisation = diar_pipeline(audio_path)

speaker_segments = []
for turn, _, speaker in diarisation.itertracks(yield_label=True):
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker,
    })

unique_speakers = set(s["speaker"] for s in speaker_segments)
logger.info(f"Detected {len(unique_speakers)} speaker(s)")
if SHOW_SPEAKER_DEBUG:
    for seg in speaker_segments:
        logger.debug(f"{seg['speaker']}: {seg['start']} - {seg['end']}")

# === LOAD WHISPER MODEL ===
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)

# === TRANSCRIBE AUDIO ===
logger.info(f"Transcribing: {INPUT_FILE}")


def assign_speaker(whisper_segment: dict[str, Any]) -> str:
    for seg in speaker_segments:
        if seg["start"] <= whisper_segment["start"] < seg["end"]:
            return seg["speaker"]
    return "Unknown"


result = model.transcribe(str(INPUT_FILE))  # type: ignore

for segment in result["segments"]:
    segment["speaker"] = assign_speaker(segment)  # type: ignore

# === SAVE .TXT OUTPUT ===
txt_path = OUTPUT_DIR / f"{base_name}.txt"
with txt_path.open("w", encoding="utf-8") as txt_file:
    txt_file.write(str(result["text"]))
logger.info(f"Text transcript saved to: {txt_path}")


def format_timestamp(seconds: float) -> str:
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


srt_path = OUTPUT_DIR / f"{base_name}.srt"
logger.info(f"Generating SRT file with {len(result['segments'])} segments")
with srt_path.open("w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(tqdm(result["segments"], desc="Writing subtitles"), start=1):
        start = format_timestamp(float(segment["start"]))  # type: ignore
        end = format_timestamp(float(segment["end"]))  # type: ignore
        text = f"{segment['speaker']}: {segment['text'].strip()}"  # type: ignore
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
logger.info(f"SRT subtitles saved to: {srt_path}")

vtt_path = OUTPUT_DIR / f"{base_name}.vtt"
logger.info("Generating WebVTT file")
with vtt_path.open("w", encoding="utf-8") as vtt_file:
    vtt_file.write("WEBVTT\n\n")
    for segment in tqdm(result["segments"], desc="Writing WebVTT"):
        start = format_vtt_timestamp(float(segment["start"]))  # type: ignore
        end = format_vtt_timestamp(float(segment["end"]))  # type: ignore
        text = f"{segment['speaker']}: {segment['text'].strip()}"  # type: ignore
        vtt_file.write(f"{start} --> {end}\n{text}\n\n")
logger.info(f"WebVTT subtitles saved to: {vtt_path}")

ttml_path = OUTPUT_DIR / f"{base_name}.ttml"
logger.info("Generating TTML (IMSC1) file")
with ttml_path.open("w", encoding="utf-8") as ttml_file:
    ttml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    ttml_file.write(
        f'<tt xml:lang="{TTML_LANGUAGE}" xmlns="http://www.w3.org/ns/ttml" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" xmlns:ttp="http://www.w3.org/ns/ttml#parameter" xmlns:tts="http://www.w3.org/ns/ttml#styling" xmlns:tt="http://www.w3.org/ns/ttml" xmlns:ims="http://www.w3.org/ns/ttml/profile/imsc1" ttp:timeBase="media" ttp:frameRate="30" ttp:profile="http://www.w3.org/ns/ttml/profile/imsc1/text">\n'
    )

    ttml_file.write("  <head>\n")
    ttml_file.write("    <metadata>\n")
    ttml_file.write(f"      <ttm:title>{xml_escape(TTML_TITLE)}</ttm:title>\n")
    ttml_file.write("    </metadata>\n")
    ttml_file.write("    <styling>\n")
    ttml_file.write('      <style xml:id="s1" tts:fontSize="100%" tts:color="white" tts:backgroundColor="black"/>\n')
    ttml_file.write("    </styling>\n")
    ttml_file.write("    <layout>\n")
    ttml_file.write(
        '      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" tts:displayAlign="after" tts:textAlign="center"/>\n'
    )
    ttml_file.write("    </layout>\n")
    ttml_file.write("  </head>\n")

    ttml_file.write("  <body>\n")
    ttml_file.write('    <div region="bottom" style="s1">\n')
    for segment in tqdm(result["segments"], desc="Writing TTML"):
        start = format_timestamp(float(segment["start"])).replace(",", ".")  # type: ignore
        end = format_timestamp(float(segment["end"])).replace(",", ".")  # type: ignore
        text = xml_escape(segment["text"].strip())  # type: ignore
        ttml_file.write(f'      <p begin="{start}" end="{end}">{text}</p>\n')
    ttml_file.write("    </div>\n")
    ttml_file.write("  </body>\n")
    ttml_file.write("</tt>\n")
logger.info(f"TTML subtitles saved to: {ttml_path}")
