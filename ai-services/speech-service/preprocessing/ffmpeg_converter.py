"""FFmpeg conversion into Whisper-compatible PCM WAV audio."""

from pathlib import Path
from typing import Optional

import ffmpeg

from config import AudioProcessingSettings
from preprocessing.exceptions import FFmpegConversionError
from preprocessing.utils import build_output_path, get_logger

logger = get_logger(__name__)


class FFmpegConverter:
    """Convert supported recordings to mono 16 kHz PCM 16-bit WAV files."""

    def __init__(self, settings: Optional[AudioProcessingSettings] = None) -> None:
        """Initialize conversion settings."""
        self._settings = settings or AudioProcessingSettings()

    def convert(self, input_path: Path) -> Path:
        """Convert a recording to the standard speech-recognition WAV format."""
        output_path = build_output_path(self._settings.processed_directory, input_path, "converted")
        try:
            (
                ffmpeg
                .input(str(input_path))
                .output(str(output_path), format="wav", acodec="pcm_s16le", ac=1, ar=self._settings.target_sample_rate)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            output_path.unlink(missing_ok=True)
            raise FFmpegConversionError("FFmpeg failed to convert the uploaded audio.") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FFmpegConversionError("FFmpeg did not produce a valid WAV output file.")
        logger.info("Audio conversion completed", extra={"input_path": str(input_path), "output_path": str(output_path)})
        return output_path

