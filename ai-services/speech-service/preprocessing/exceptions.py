"""Custom exceptions raised by the audio preprocessing pipeline."""


class AudioProcessingError(Exception):
    """Base class for all speech preprocessing failures."""


class UnsupportedFormatError(AudioProcessingError):
    """Raised when an audio file extension or MIME type is unsupported."""


class AudioTooLargeError(AudioProcessingError):
    """Raised when an audio file exceeds the configured maximum size."""


class CorruptedAudioError(AudioProcessingError):
    """Raised when audio metadata cannot be read or is invalid."""


class InvalidSampleRateError(AudioProcessingError):
    """Raised when an input sample rate is outside the accepted range."""


class RecordingTooLongError(AudioProcessingError):
    """Raised when an audio recording exceeds the configured duration limit."""


class FFmpegConversionError(AudioProcessingError):
    """Raised when FFmpeg cannot convert an input file."""


class NormalizationError(AudioProcessingError):
    """Raised when loudness or silence normalization fails."""


class VADProcessingError(AudioProcessingError):
    """Raised when voice activity detection cannot produce speech audio."""

