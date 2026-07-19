"""Thread-safe, production-safe loader for a singleton Faster-Whisper model."""

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from preprocessing.utils import get_logger
from whisper.exceptions import ModelLoadError

logger = get_logger(__name__)
ModelFactory = Callable[..., Any]


@dataclass(frozen=True)
class WhisperSettings:
    """Faster-Whisper settings sourced from environment variables."""

    model_size: str
    device: str
    compute_type: str
    download_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "models" / "faster-whisper"
    )

    @classmethod
    def from_environment(cls) -> "WhisperSettings":
        """Load and validate the model runtime configuration from the environment."""
        model_size = os.getenv("WHISPER_MODEL", "small").strip()
        if not model_size:
            raise ModelLoadError("WHISPER_MODEL must name a Faster-Whisper model or local model directory.")
        requested_device = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ModelLoadError("WHISPER_DEVICE must be one of: auto, cpu, cuda.")
        device = cls._detect_device() if requested_device == "auto" else requested_device
        configured_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()
        compute_type = cls._resolve_compute_type(device, configured_compute_type or None)
        cache_value = os.getenv("WHISPER_DOWNLOAD_ROOT", "").strip()
        download_root = (
            Path(cache_value).expanduser().resolve()
            if cache_value
            else Path(__file__).resolve().parents[3] / "data" / "models" / "faster-whisper"
        )
        return cls(model_size, device, compute_type, download_root)

    @staticmethod
    def _detect_device() -> str:
        """Use CUDA only when the installed CTranslate2 runtime exposes a GPU."""
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except (ImportError, AttributeError):
            return "cpu"

    @staticmethod
    def _resolve_compute_type(device: str, requested: Optional[str]) -> str:
        """Use a compute type supported by the active CTranslate2 device."""
        preferred = requested or ("float16" if device == "cuda" else "int8")
        try:
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types(device)
        except (ImportError, AttributeError):
            return preferred
        if preferred in supported:
            return preferred
        fallback = "int8" if device == "cpu" and "int8" in supported else "float32"
        if fallback not in supported:
            raise ModelLoadError(
                "CTranslate2 does not support compute type '{0}' on {1}. Supported: {2}.".format(
                    preferred, device, ", ".join(sorted(supported))
                )
            )
        logger.warning(
            "Requested compute type is unsupported; using fallback %s for %s",
            fallback,
            device,
        )
        return fallback


class FasterWhisperModelLoader:
    """Load and retain exactly one configured Faster-Whisper model instance."""

    _instance: Optional["FasterWhisperModelLoader"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "FasterWhisperModelLoader":
        """Create one loader object in a thread-safe manner."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: Optional[WhisperSettings] = None, model_factory: Optional[ModelFactory] = None) -> None:
        """Initialize configuration without loading the model eagerly."""
        if getattr(self, "_initialized", False):
            return
        self.settings = settings or WhisperSettings.from_environment()
        self._model_factory = model_factory or self._default_model_factory
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._initialized = True

    def get_model(self) -> Any:
        """Lazily download if needed, load, and cache the Faster-Whisper model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        self._prepare_cache_directory()
                        logger.info(
                            "Loading Faster Whisper model name=%s device=%s compute_type=%s cache=%s",
                            self.settings.model_size,
                            self.settings.device,
                            self.settings.compute_type,
                            self.settings.download_root,
                        )
                        self._model = self._model_factory(
                            self.settings.model_size,
                            device=self.settings.device,
                            compute_type=self.settings.compute_type,
                            download_root=str(self.settings.download_root),
                        )
                    except Exception as exc:
                        reason = self._describe_failure(exc)
                        logger.exception(
                            "Model loading failed name=%s device=%s compute_type=%s cache=%s reason=%s",
                            self.settings.model_size,
                            self.settings.device,
                            self.settings.compute_type,
                            self.settings.download_root,
                            reason,
                        )
                        if isinstance(exc, ModelLoadError):
                            raise
                        raise ModelLoadError(
                            "Unable to load Faster-Whisper model '{0}' on {1} with {2}. {3}".format(
                                self.settings.model_size,
                                self.settings.device,
                                self.settings.compute_type,
                                reason,
                            )
                        ) from exc
        return self._model

    def _prepare_cache_directory(self) -> None:
        """Create and validate the model download directory before model loading."""
        try:
            self.settings.download_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelLoadError(
                "Cannot create Whisper model cache directory '{0}': {1}".format(
                    self.settings.download_root, exc
                )
            ) from exc
        if not self.settings.download_root.is_dir():
            raise ModelLoadError(
                "Whisper model cache path is not a directory: {0}".format(self.settings.download_root)
            )

    @staticmethod
    def _describe_failure(error: Exception) -> str:
        """Describe common package, network, cache, and model selection failures."""
        message = str(error).strip() or repr(error)
        lowered = message.lower()
        if isinstance(error, ImportError):
            return "Missing dependency: {0}".format(message)
        if any(token in lowered for token in ("network", "connection", "timeout", "ssl", "download")):
            return "Model download/network failure: {0}".format(message)
        if any(token in lowered for token in ("not found", "repository", "model")):
            return "Invalid model name or unavailable model cache: {0}".format(message)
        if any(token in lowered for token in ("corrupt", "checksum", "invalid archive")):
            return "Corrupted model cache: {0}".format(message)
        return "{0}: {1}".format(type(error).__name__, message)

    @staticmethod
    def _default_model_factory(*args: Any, **kwargs: Any) -> Any:
        """Import Faster-Whisper only when inference is first requested."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError("Missing dependency: install faster-whisper and ctranslate2.") from exc
        return WhisperModel(*args, **kwargs)

    @classmethod
    def reset_for_tests(cls) -> None:
        """Clear singleton state for isolated unit tests only."""
        with cls._instance_lock:
            cls._instance = None