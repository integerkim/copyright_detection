"""Media preprocessing only: decode video/audio input to analysis-ready WAV."""
import shutil
import subprocess
import sys
from pathlib import Path

def extract_audio(source: Path, target: Path, sample_rate: int = 22050, channels: int = 1) -> Path:
    """Use FFmpeg to extract/convert media into PCM WAV; no features are computed here."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sibling = Path(sys.executable).resolve().parent / "ffmpeg"
        ffmpeg = str(sibling) if sibling.is_file() else None
    if not ffmpeg:
        if source.suffix.lower() == ".wav":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return target
        raise RuntimeError("FFmpeg is required to extract audio from video, but it is not installed")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-v", "error", "-i", str(source), "-vn", "-ac", str(channels),
               "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr.strip()}")
    return target

def preprocess(source: Path, target: Path, sample_rate: int = 22050, channels: int = 1) -> Path:
    return extract_audio(source, target, sample_rate, channels)
