"""Unit tests for thread-safe lazy Faster-Whisper model loading."""

from pathlib import Path

from whisper.model import FasterWhisperModelLoader, WhisperSettings


def test_model_is_loaded_lazily_and_once() -> None:
    """Do not invoke the factory until get_model and cache its result."""
    FasterWhisperModelLoader.reset_for_tests()
    calls = []

    def factory(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    loader = FasterWhisperModelLoader(
        WhisperSettings("small", "cpu", "int8"), model_factory=factory
    )
    assert calls == []
    assert loader.get_model() is loader.get_model()
    assert len(calls) == 1


def test_auto_device_configuration_has_valid_values(monkeypatch: object) -> None:
    """Load model configuration from its supported environment variables."""
    FasterWhisperModelLoader.reset_for_tests()
    settings = WhisperSettings.from_environment()
    assert settings.model_size
    assert settings.device in {"cpu", "cuda"}


def test_model_loader_passes_cache_directory_to_factory(tmp_path: Path) -> None:
    """Provide Faster-Whisper a writable cache directory for automatic downloads."""
    FasterWhisperModelLoader.reset_for_tests()
    captured = {}

    def factory(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    settings = WhisperSettings("small", "cpu", "int8", tmp_path / "model-cache")
    FasterWhisperModelLoader(settings, factory).get_model()
    assert Path(captured["download_root"]).is_dir()