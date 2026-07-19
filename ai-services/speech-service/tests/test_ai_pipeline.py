"""Tests for central AI pipeline data flow."""

from pathlib import Path
from types import SimpleNamespace

from pipeline.ai_pipeline import AIPipeline
from pipeline.preprocessing_pipeline import PreprocessingResult
from whisper.transcript_storage import StoredTranscript


class RecordingPreprocessingPipeline:
    """Preprocessing boundary that exposes the exact VAD hand-off path."""

    def __init__(self, result: PreprocessingResult) -> None:
        self.result = result
        self.received_input: Path | None = None

    def process(self, input_audio_path: str | Path, content_type: str | None = None) -> PreprocessingResult:
        """Return the staged preprocessing output."""
        self.received_input = Path(input_audio_path)
        return self.result


class RecordingTranscriptionService:
    """ASR boundary used to verify only VAD output reaches transcription."""

    def __init__(self) -> None:
        self.received_audio: Path | None = None

    def transcribe(self, meeting_id: str, audio_path: str | Path) -> object:
        """Return a transcript-shaped object for the orchestration boundary."""
        self.received_audio = Path(audio_path)
        return SimpleNamespace(
            language="en",
            transcript="Meeting started",
            segments=[],
            metadata=SimpleNamespace(model_dump=lambda mode: {"model_name": "faster-whisper"}),
        )


class RecordingTranscriptStorage:
    """Persistence boundary that avoids filesystem coupling in orchestration tests."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.saved_name: str | None = None

    def save(self, meeting_name: str, transcript: object) -> StoredTranscript:
        """Return deterministic storage paths for a completed transcript."""
        self.saved_name = meeting_name
        return StoredTranscript(self.output_directory / "meeting_transcript.json", self.output_directory / "meeting_transcript.txt")


def test_ai_pipeline_passes_only_vad_output_to_asr(tmp_path: Path) -> None:
    """Ensure Whisper receives the VAD path rather than the original recording."""
    staged = PreprocessingResult(
        original_audio=tmp_path / "meeting.m4a",
        wav_audio=tmp_path / "converted.wav",
        normalized_audio=tmp_path / "normalized.wav",
        vad_audio=tmp_path / "vad.wav",
        duration=12.0,
        sample_rate=16_000,
        processing_time=0.2,
    )
    preprocessing = RecordingPreprocessingPipeline(staged)
    transcription = RecordingTranscriptionService()
    storage = RecordingTranscriptStorage(tmp_path)
    result = AIPipeline(preprocessing, transcription, storage).process("meeting-1", staged.original_audio)
    assert transcription.received_audio == staged.vad_audio
    assert result.transcript == "Meeting started"
    assert result.transcript_json.endswith("meeting_transcript.json")
    assert storage.saved_name == "meeting"
