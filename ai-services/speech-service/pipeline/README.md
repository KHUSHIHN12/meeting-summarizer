# Speech Service Pipeline

## Overview

The `pipeline` package is the central orchestration layer for the Speech Service. It coordinates existing, single-responsibility components without reimplementing validation, audio conversion, normalization, VAD, or Whisper ASR logic.

## Architecture

```text
Meeting Audio
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
Whisper ASR
    │
    ▼
Transcript Builder
    │
    ▼
Structured Transcript
```

## Pipeline Files

- `preprocessing_pipeline.py`: passes typed hand-off data through validation, FFmpeg conversion, normalization, and VAD.
- `ai_pipeline.py`: invokes preprocessing, then sends only the VAD WAV output to the existing Whisper transcription service.
- `pipeline_runner.py`: the only executable pipeline file; it handles CLI input, JSON output, timing, and exit codes.

## Execution

Run from `ai-services/speech-service`:

```bash
python3 pipeline/pipeline_runner.py ../../../data/uploads/sample_meeting.m4a \
  --meeting-id meeting-001 \
  --content-type audio/mp4
```

## Expected JSON Response

```json
{
  "success": true,
  "original_audio": ".../sample_meeting.m4a",
  "wav_audio": ".../data/processed/..._converted_....wav",
  "normalized_audio": ".../data/processed/..._normalized_....wav",
  "vad_audio": ".../data/processed/..._vad_....wav",
  "language": "en",
  "transcript": "...",
  "segments": [],
  "metadata": {},
  "processing_time": 0.0
}
```

## Data Flow

Every stage receives the output of the immediately preceding stage. The original meeting recording is read only by validation and conversion; normalization receives the converted WAV, VAD receives normalized WAV, and Whisper receives the VAD WAV. A stage failure stops the pipeline immediately.

## Extending the Pipeline

Future stages are appended in `AIPipeline` after the current transcript hand-off. Speaker diarization, LangGraph workflow nodes, summarization, action-item extraction, meeting minutes, and document export can each be added as a new injected stage without changing existing stage implementations.
## Transcript Output

After Whisper ASR and transcript building succeed, the pipeline automatically persists two files beneath the repository root:

```text
meeting-summarizer/
└── data/
    └── transcripts/
        ├── sample_meeting_transcript.json
        └── sample_meeting_transcript.txt
```

- **JSON transcript**: contains meeting name, language, duration, processing time, complete transcript, timestamped segments, confidence values, and model metadata. It is designed for APIs and downstream services.
- **TXT transcript**: contains a readable meeting header followed by the complete transcript for human review.

The AI pipeline returns `transcript_json` and `transcript_text` paths with the normal transcript response. The folder is created automatically when the first transcript is saved.
