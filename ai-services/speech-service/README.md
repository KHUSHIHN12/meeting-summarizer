# Speech Preprocessing Service

## Service Overview

The Speech Preprocessing Service prepares uploaded meeting recordings for a downstream Whisper ASR service. It is an isolated preprocessing component and does not perform transcription, speaker diarization, summarization, or action-item extraction.

## Responsibilities

- Validate supported audio uploads and their metadata.
- Convert recordings to 16 kHz, mono, 16-bit PCM WAV.
- Normalize loudness, trim boundary silence, and reduce clipping.
- Detect speech with WebRTC VAD, produce speech timestamps, and remove long silence.
- Store processed WAV files in `data/processed/`.

## Folder Structure

```text
speech-service/
├── config.py
├── preprocessing/
│   ├── audio_processor.py
│   ├── exceptions.py
│   ├── ffmpeg_converter.py
│   ├── normalizer.py
│   ├── utils.py
│   ├── vad.py
│   └── validator.py
├── requirements.txt
└── tests/
```

## Processing Pipeline

```text
Uploaded Audio
      │
      ▼
Validation
      │
      ▼
FFmpeg Conversion
      │
      ▼
Normalization
      │
      ▼
Voice Activity Detection
      │
      ▼
Processed Audio
      │
      ▼
Whisper ASR

```

1. Audio Validation

Validate:

Supported formats:
.mp3
.wav
.m4a
.mp4
Sample rate
Duration
Empty or corrupted files
Maximum file size
2. FFmpeg Integration

Use FFmpeg to convert every uploaded file into a standard format.

Target format:

WAV
Mono channel
16 kHz sample rate
PCM 16-bit encoding

This is the format recommended for Whisper.

Example conversion:

meeting.mp3
      │
      ▼
meeting_processed.wav
3. Audio Normalization

Normalize audio volume to improve transcription quality.

Tasks:

Remove volume inconsistencies
Normalize loudness
Trim silence at the beginning and end

4. Voice Activity Detection (VAD)

Use a VAD model (e.g., Silero VAD or WebRTC VAD) to detect speech and remove long silent segments.

Example:

Original Audio

----silence----
Hello everyone
----silence----
Discussion
----silence----
Meeting ended

↓

Processed Audio

Hello everyone
Discussion
Meeting ended


## Technologies Used

- Python 3.10+
- FFmpeg and `ffmpeg-python`
- `pydub`, `numpy`, `soundfile`, and `librosa`
- WebRTC VAD (`webrtcvad`)
- `pytest`

## Installation

FFmpeg must be installed and available on your `PATH`.

```powershell
cd ai-services/speech-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependencies

The runtime dependencies are listed in `requirements.txt`. WebRTC VAD accepts 16-bit PCM frames at supported sample rates; the converter guarantees 16 kHz mono PCM input before VAD runs.

## Configuration

`AudioProcessingSettings` in `config.py` controls file size and duration limits, accepted sample-rate range, the target WAV format, VAD aggressiveness, and the output directory. By default processed files are written to `data/processed/` at the repository root.

## Running Tests

```powershell
cd ai-services/speech-service
pytest tests -q
```

The FFmpeg integration test is skipped automatically when the FFmpeg binary is not installed.

## Future Enhancements

- Add configurable cloud/object storage for processed recordings.
- Add metrics and distributed tracing.
- Add a queue consumer for asynchronous processing.
- Evaluate Silero VAD as an alternative VAD engine.
- Add audio quality scoring and noise-reduction stages.
## Pipeline Architecture

`AudioPreprocessingPipeline` composes the existing validation, conversion, normalization, and VAD components without duplicating their responsibilities. Each stage is independently replaceable and can later be represented as a separate LangGraph node.

```text
AudioPreprocessingPipeline
├── AudioValidator
├── FFmpegConverter
├── AudioNormalizer
└── VoiceActivityDetector
```

## Execution Flow

```text
Input Audio
    │
    ▼
Audio Validation
    │ original source path
    ▼
FFmpeg Conversion
    │ converted WAV path
    ▼
Audio Normalization
    │ normalized WAV path
    ▼
Voice Activity Detection
    │ speech-only WAV path
    ▼
Processed Audio
```

A failure at any stage stops execution immediately, logs the failure, and propagates the originating informative exception. Intermediate outputs are written to `data/processed/` by the underlying stages.

## CLI Example

Run from `ai-services/speech-service` after activating its environment:

```powershell
python preprocessing/pipeline.py ../../../data/uploads/project_meeting.mp3
```

Optionally validate the MIME type supplied by an upstream upload service:

```powershell
python preprocessing/pipeline.py ../../../data/uploads/project_meeting.mp3 --content-type audio/mpeg
```

## Expected Output

The CLI reports progress for each successful stage and then prints a structured result:

```python
{
    "success": True,
    "original_audio": ".../project_meeting.mp3",
    "wav_audio": ".../data/processed/project_meeting_converted_<id>.wav",
    "normalized_audio": ".../data/processed/project_meeting_converted_<id>_normalized_<id>.wav",
    "vad_audio": ".../data/processed/project_meeting_converted_<id>_normalized_<id>_vad_<id>.wav",
    "processing_time": 1.2345,
    "sample_rate": 44100,
    "duration": 126.5,
}
```

## Output Folder Structure

```text
data/
└── processed/
    ├── <recording>_converted_<id>.wav
    ├── <recording>_normalized_<id>.wav
    └── <recording>_vad_<id>.wav
```

## Troubleshooting

- **`FFmpeg is unavailable`**: install FFmpeg in WSL/Linux and ensure `ffmpeg` and `ffprobe` are available on `PATH`.
- **No speech detected**: confirm the recording contains audible speech; silence-only recordings deliberately raise `VADProcessingError`.
- **Unsupported MIME type**: pass the MIME type emitted by the upload client, such as `audio/mpeg`, `audio/wav`, `audio/m4a`, or `video/mp4`.
- **Import errors**: activate the speech-service virtual environment and run `pip install -r requirements.txt`.
