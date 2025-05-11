import logging

# === CONFIGURATION ===
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import whisper
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "../.local/test.mp4"
OUTPUT_DIR = SCRIPT_DIR / "../.local"
MODEL_SIZE = "medium"  # tiny, base, small, medium, large

# === LOGGING SETUP ===
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# === PATH RESOLUTION & SAFETY ===
INPUT_FILE = INPUT_FILE.resolve()
OUTPUT_DIR = OUTPUT_DIR.resolve()

if not INPUT_FILE.exists():
    logger.error(f"Input file not found: {INPUT_FILE}")
    exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
base_name = INPUT_FILE.stem  # Filename without extension

# === LOAD WHISPER MODEL ===
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)

# === TRANSCRIBE AUDIO ===
logger.info(f"Transcribing: {INPUT_FILE}")
result = model.transcribe(str(INPUT_FILE))

# === SAVE .TXT OUTPUT ===
txt_path = OUTPUT_DIR / f"{base_name}.txt"
with txt_path.open("w", encoding="utf-8") as txt_file:
    txt_file.write(result["text"])
logger.info(f"Text transcript saved to: {txt_path}")


# === SRT HELPER FUNCTION ===
def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (hh:mm:ss,ms)."""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format timestamp for WebVTT (hh:mm:ss.mmm)"""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


# === SAVE .SRT OUTPUT WITH PROGRESS BAR ===
srt_path = OUTPUT_DIR / f"{base_name}.srt"
logger.info(f"Generating SRT file with {len(result['segments'])} segments")

with srt_path.open("w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(tqdm(result["segments"], desc="Writing subtitles"), start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")

logger.info(f"SRT subtitles saved to: {srt_path}")

# === GENERATE WEBVTT (.vtt) FILE ===
vtt_path = OUTPUT_DIR / f"{base_name}.vtt"
logger.info("Generating WebVTT file")

with vtt_path.open("w", encoding="utf-8") as vtt_file:
    vtt_file.write("WEBVTT\n\n")
    for i, segment in enumerate(tqdm(result["segments"], desc="Writing WebVTT"), start=1):
        start = format_vtt_timestamp(segment["start"])
        end = format_vtt_timestamp(segment["end"])
        text = segment["text"].strip()
        vtt_file.write(f"{start} --> {end}\n{text}\n\n")

logger.info(f"WebVTT subtitles saved to: {vtt_path}")

# === GENERATE TTML / IMSC1 FILE (WITH HEAD, VALIDATED STRUCTURE) ===
ttml_path = OUTPUT_DIR / f"{base_name}.ttml"
logger.info("Generating TTML (IMSC1) file")

with ttml_path.open("w", encoding="utf-8") as ttml_file:
    ttml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    ttml_file.write(
        '<tt xml:lang="en"\n'
        '    xmlns="http://www.w3.org/ns/ttml"\n'
        '    xmlns:ttm="http://www.w3.org/ns/ttml#metadata"\n'
        '    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"\n'
        '    xmlns:tts="http://www.w3.org/ns/ttml#styling"\n'
        '    xmlns:tt="http://www.w3.org/ns/ttml"\n'
        '    ttp:timeBase="media"\n'
        '    ttp:frameRate="30">\n'
    )

    # --- Head section with basic styling and layout
    ttml_file.write("  <head>\n")
    ttml_file.write("    <styling>\n")
    ttml_file.write('      <style xml:id="s1" tts:fontSize="100%" tts:color="white" tts:backgroundColor="black"/>\n')
    ttml_file.write("    </styling>\n")
    ttml_file.write("    <layout>\n")
    ttml_file.write(
        '      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" '
        'tts:displayAlign="after" tts:textAlign="center"/>\n'
    )
    ttml_file.write("    </layout>\n")
    ttml_file.write("  </head>\n")

    # --- Body content
    ttml_file.write("  <body>\n")
    ttml_file.write('    <div region="bottom" style="s1">\n')

    for i, segment in enumerate(tqdm(result["segments"], desc="Writing TTML"), start=1):
        start = format_timestamp(segment["start"]).replace(",", ".")
        end = format_timestamp(segment["end"]).replace(",", ".")
        text = xml_escape(segment["text"].strip())
        ttml_file.write(f'      <p begin="{start}" end="{end}">{text}</p>\n')

    ttml_file.write("    </div>\n")
    ttml_file.write("  </body>\n")
    ttml_file.write("</tt>\n")

logger.info(f"TTML subtitles saved to: {ttml_path}")
