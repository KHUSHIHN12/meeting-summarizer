"""WebRTC voice activity detection and silence removal."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
import webrtcvad

from config import AudioProcessingSettings
from preprocessing.exceptions import VADProcessingError
from preprocessing.utils import build_output_path, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeechRegion:
    """A contiguous speech range represented in seconds."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class VADResult:
    """Result of VAD processing and resulting cleaned audio path."""

    output_path: Path
    speech_regions: list[SpeechRegion]


class VoiceActivityDetector:
    """Detect speech frames with WebRTC VAD and remove long silences."""

    def __init__(self, settings: Optional[AudioProcessingSettings] = None) -> None:
        """Initialize WebRTC VAD configuration."""
        self._settings = settings or AudioProcessingSettings()
        self._vad = webrtcvad.Vad(self._settings.vad_aggressiveness)

    def detect(self, input_path: Path) -> VADResult:
        """Detect speech regions and export an audio file containing speech only."""
        output_path = build_output_path(self._settings.processed_directory, input_path, "vad")
        try:
            audio = AudioSegment.from_wav(input_path).set_channels(1).set_frame_rate(self._settings.target_sample_rate).set_sample_width(2)
            frame_bytes = int(audio.frame_rate * self._settings.vad_frame_duration_ms / 1000) * audio.sample_width
            regions = self._detect_regions(audio.raw_data, audio.frame_rate, frame_bytes)
            if not regions:
                raise VADProcessingError("No speech was detected in the audio recording.")
            cleaned_audio = sum((audio[int(region.start_seconds * 1000):int(region.end_seconds * 1000)] for region in regions), AudioSegment.silent(duration=0, frame_rate=audio.frame_rate))
            cleaned_audio.export(output_path, format="wav")
        except VADProcessingError:
            output_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise VADProcessingError("Voice activity detection failed.") from exc
        logger.info("Voice activity detection completed", extra={"input_path": str(input_path), "output_path": str(output_path), "duration_seconds": len(cleaned_audio) / 1000})
        return VADResult(output_path=output_path, speech_regions=regions)

    def _detect_regions(self, raw_audio: bytes, sample_rate: int, frame_bytes: int) -> list[SpeechRegion]:
        """Classify fixed-size audio frames and merge nearby speech regions."""
        frame_duration = self._settings.vad_frame_duration_ms / 1000
        speech_ranges: list[SpeechRegion] = []
        current_start: Optional[float] = None
        last_speech_end: Optional[float] = None
        for offset in range(0, len(raw_audio) - frame_bytes + 1, frame_bytes):
            start = (offset // frame_bytes) * frame_duration
            is_speech = self._vad.is_speech(raw_audio[offset:offset + frame_bytes], sample_rate)
            if is_speech:
                if current_start is None:
                    current_start = start
                last_speech_end = start + frame_duration
            elif current_start is not None and last_speech_end is not None and (start - last_speech_end) * 1000 > self._settings.max_silence_between_speech_ms:
                speech_ranges.append(SpeechRegion(current_start, last_speech_end))
                current_start, last_speech_end = None, None
        if current_start is not None and last_speech_end is not None:
            speech_ranges.append(SpeechRegion(current_start, last_speech_end))
        return speech_ranges

