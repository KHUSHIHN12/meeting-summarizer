"""Service that transcribes VAD-processed WAV audio using Faster-Whisper."""

from math import exp
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import soundfile as sf

from whisper.exceptions import AudioNotFoundError, ModelLoadError, TranscriptionError, UnsupportedAudioError
from whisper.model import FasterWhisperModelLoader
from whisper.schemas import TranscriptResponse, TranscriptSegment
from whisper.transcript_builder import TranscriptBuilder
from whisper.utils import normalize_language
from preprocessing.utils import get_logger

logger = get_logger(__name__)


class TranscriptionService:
    """Transcribe validated VAD WAV files through an injected model loader."""

    def __init__(
        self,
        model_loader: Optional[FasterWhisperModelLoader] = None,
        transcript_builder: Optional[TranscriptBuilder] = None,
    ) -> None:
        """Initialize transcription dependencies for ASR inference."""
        self._model_loader = model_loader or FasterWhisperModelLoader()
        self._builder = transcript_builder or TranscriptBuilder()

    def transcribe(self, meeting_id: str, audio_path: str | Path) -> TranscriptResponse:
        """Transcribe a VAD-processed WAV file into timestamped text.

        Args:
            meeting_id: Identifier assigned by the meeting service.
            audio_path: VAD-produced mono PCM WAV file path.
        """
        path = Path(audio_path).expanduser().resolve()
        duration = self._validate_vad_audio(path)
        started_at = perf_counter()
        logger.info("Starting transcription", extra={"input_path": str(path)})
        try:
            segments_source, information = self._model_loader.get_model().transcribe(
                str(path), vad_filter=False
            )
            language = normalize_language(getattr(information, "language", "unknown"))
            segments = self._map_segments(segments_source)
        except ModelLoadError:
            logger.exception("Whisper model loading failed", extra={"input_path": str(path)})
            raise
        except Exception as exc:
            logger.exception("Transcription inference failed", extra={"input_path": str(path)})
            raise TranscriptionError("Faster-Whisper transcription failed: {0}".format(exc)) from exc

        processing_time = perf_counter() - started_at
        logger.info("Detected language: {0}".format(language))
        logger.info("Generated {0} transcript segments".format(len(segments)))
        response = self._builder.build(
            meeting_id=meeting_id,
            language=language,
            duration=duration,
            segments=segments,
            processing_time=processing_time,
            model_size=self._model_loader.settings.model_size,
        )
        logger.info("Transcript completed", extra={"input_path": str(path), "elapsed_ms": round(processing_time * 1000, 2)})
        return response

    @staticmethod
    def _validate_vad_audio(path: Path) -> float:
        """Ensure input exists, is WAV, and contains readable audio frames."""
        if not path.is_file():
            raise AudioNotFoundError("The VAD-processed audio file was not found.")
        if path.suffix.lower() != ".wav":
            raise UnsupportedAudioError("Whisper ASR accepts only VAD-processed WAV audio.")
        try:
            info = sf.info(path)
        except RuntimeError as exc:
            raise TranscriptionError("The VAD-processed WAV file is corrupted.") from exc
        if info.samplerate <= 0 or info.frames <= 0:
            raise TranscriptionError("The VAD-processed WAV file is empty or invalid.")
        return float(info.frames) / info.samplerate

    @staticmethod
    def _map_segments(segments_source: Any) -> list[TranscriptSegment]:
        """Map Faster-Whisper segment objects into stable response schemas."""
        mapped_segments = []
        for fallback_id, segment in enumerate(segments_source):
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            confidence = getattr(segment, "avg_logprob", None)
            mapped_segments.append(
                TranscriptSegment(
                    segment_id=int(getattr(segment, "id", fallback_id)),
                    start_time=float(getattr(segment, "start", 0.0)),
                    end_time=float(getattr(segment, "end", 0.0)),
                    text=text,
                    confidence=max(0.0, min(1.0, exp(float(confidence)))) if confidence is not None else None,
                )
            )
        return mapped_segments
