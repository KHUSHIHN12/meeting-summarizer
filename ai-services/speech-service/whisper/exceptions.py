"""Exceptions specific to Faster-Whisper transcription."""


class WhisperServiceError(Exception):
    """Base exception for the speech transcription service."""


class ModelLoadError(WhisperServiceError):
    """Raised when the Faster-Whisper model cannot be initialized."""


class AudioNotFoundError(WhisperServiceError):
    """Raised when a requested VAD-processed WAV file is absent."""


class UnsupportedAudioError(WhisperServiceError):
    """Raised when transcription input is not a supported WAV recording."""


class TranscriptionError(WhisperServiceError):
    """Raised when audio decoding or ASR inference fails."""

