"""Central end-to-end AI orchestration for the speech service."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from pipeline.preprocessing_pipeline import PreprocessingPipeline, PreprocessingResult
from preprocessing.utils import get_logger
from whisper.transcriber import TranscriptionService
from whisper.transcript_storage import TranscriptStorage

logger = get_logger(__name__)


@dataclass(frozen=True)
class AIPipelineResult:
    """Complete speech-service result produced from one meeting recording."""

    success: bool
    original_audio: str
    wav_audio: str
    normalized_audio: str
    vad_audio: str
    transcript_json: str
    transcript_text: str
    language: str
    transcript: str
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    processing_time: float

    def to_dict(self) -> dict[str, Any]:
        """Return the response as a JSON-serializable dictionary."""
        return asdict(self)


class AIPipeline:
    """Compose preprocessing, ASR, and transcript persistence into one flow."""

    def __init__(
        self,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        transcription_service: Optional[TranscriptionService] = None,
        transcript_storage: Optional[TranscriptStorage] = None,
    ) -> None:
        """Inject stages so future graph nodes remain independently replaceable."""
        self._preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline()
        self._transcription_service = transcription_service or TranscriptionService()
        self._transcript_storage = transcript_storage or TranscriptStorage()

    def process(
        self,
        meeting_id: str,
        input_audio_path: str | Path,
        content_type: Optional[str] = None,
    ) -> AIPipelineResult:
        """Run preprocessing, ASR, then persist the structured transcript output."""
        started_at = perf_counter()
        source_path = Path(input_audio_path).expanduser().resolve()
        logger.info("Starting AI Pipeline", extra={"input_path": str(source_path)})
        try:
            preprocessing_result = self._preprocessing_pipeline.process(source_path, content_type)
            transcript = self._transcription_service.transcribe(meeting_id, preprocessing_result.vad_audio)
            stored_transcript = self._transcript_storage.save(source_path.stem, transcript)
        except Exception:
            logger.exception("AI Pipeline failed", extra={"input_path": str(source_path)})
            raise

        result = self._build_result(
            preprocessing_result,
            transcript,
            stored_transcript.json_path,
            stored_transcript.text_path,
            perf_counter() - started_at,
        )
        logger.info(
            "AI Pipeline completed successfully",
            extra={"output_path": result.transcript_json, "elapsed_ms": round(result.processing_time * 1000, 2)},
        )
        return result

    @staticmethod
    def _build_result(
        preprocessing_result: PreprocessingResult,
        transcript: Any,
        transcript_json: Path,
        transcript_text: Path,
        processing_time: float,
    ) -> AIPipelineResult:
        """Combine stage hand-offs and stored paths into the final response."""
        return AIPipelineResult(
            success=True,
            original_audio=str(preprocessing_result.original_audio),
            wav_audio=str(preprocessing_result.wav_audio),
            normalized_audio=str(preprocessing_result.normalized_audio),
            vad_audio=str(preprocessing_result.vad_audio),
            transcript_json=str(transcript_json),
            transcript_text=str(transcript_text),
            language=transcript.language,
            transcript=transcript.transcript,
            segments=[segment.model_dump(mode="json") for segment in transcript.segments],
            metadata=transcript.metadata.model_dump(mode="json"),
            processing_time=round(processing_time, 4),
        )