# Whisper ASR Service

## Overview

The **Whisper ASR (Automatic Speech Recognition)** module is responsible for converting preprocessed meeting audio into accurate text transcripts. It is the second major AI stage in the Meeting Summarizer pipeline, following Audio Preprocessing.

This module uses **Faster-Whisper**, an optimized implementation of OpenAI's Whisper model built on **CTranslate2**, providing significantly faster inference with lower memory consumption while maintaining nearly identical transcription accuracy.

The primary objective of this service is to transform spoken meeting conversations into structured textual data that can be consumed by downstream AI components such as Speaker Diarization, LangGraph workflow orchestration, Meeting Summarization, and Action Item Extraction.

---

# Objectives

The Whisper ASR module performs the following tasks:

- Convert speech into text.
- Detect the spoken language automatically.
- Generate timestamped transcript segments.
- Build a complete meeting transcript.
- Produce structured transcript objects.
- Prepare transcript data for MongoDB storage.
- Supply transcript data to downstream AI modules.

---

# Architecture

```
                Meeting Audio
                      │
                      ▼
          Audio Preprocessing Pipeline
      (Validation → FFmpeg → Normalize → VAD)
                      │
                      ▼
                Whisper ASR
                      │
                      ▼
             Transcript Builder
                      │
                      ▼
           Structured Transcript
                      │
                      ▼
             MongoDB Storage
                (Future Phase)
                      │
                      ▼
          Speaker Diarization
                (Future Phase)
                      │
                      ▼
           Meeting Summarization
                (Future Phase)
                      │
                      ▼
        Action Item Extraction
                (Future Phase)
```

---

# Folder Structure

```
whisper/
│
├── __init__.py
├── model.py
├── transcriber.py
├── transcript_builder.py
├── schemas.py
├── utils.py
├── exceptions.py
└── README.md
```

## File Description

| File | Responsibility |
|------|----------------|
| model.py | Loads and manages the Faster-Whisper model. |
| transcriber.py | Performs speech-to-text transcription. |
| transcript_builder.py | Creates the final structured transcript object. |
| schemas.py | Pydantic models for transcript data. |
| utils.py | Helper functions such as timestamp formatting and transcript merging. |
| exceptions.py | Custom exception classes. |
| README.md | Module documentation. |

---

# Workflow

The complete transcription workflow is illustrated below.

```
Input Audio (VAD Output)
        │
        ▼
Load Faster Whisper Model
        │
        ▼
Speech Recognition
        │
        ▼
Language Detection
        │
        ▼
Generate Timestamped Segments
        │
        ▼
Merge Transcript Segments
        │
        ▼
Build Transcript Object
        │
        ▼
Return Structured Response
```

---

# What is Whisper ASR?

Whisper is an Automatic Speech Recognition (ASR) model developed by OpenAI that converts spoken language into text.

It supports:

- Speech-to-text transcription
- Automatic language detection
- Multilingual transcription
- Timestamp generation

In this project, Whisper serves as the core transcription engine.

---

# Why Faster-Whisper?

Instead of using the original Whisper implementation, this project uses **Faster-Whisper** because it offers several production advantages.

| Feature | Whisper | Faster-Whisper |
|----------|----------|----------------|
| Accuracy | Excellent | Excellent |
| Speed | Standard | Faster |
| Memory Usage | High | Lower |
| Backend | PyTorch | CTranslate2 |
| CPU Performance | Moderate | Optimized |
| GPU Support | Yes | Yes |

Benefits include:

- Faster transcription
- Lower RAM usage
- Better deployment performance
- Scalable for production environments

---

# Supported Model Sizes

| Model | Speed | Accuracy | Recommended Use |
|--------|---------|------------|----------------|
| tiny | Very Fast | Basic | Testing |
| base | Fast | Good | Small applications |
| small | Fast | High | **used** |
| medium | Moderate | Higher | Large meetings |
| large-v3 | Slower | Best | Production with GPU |

Default model:

```
small
```

---

# Transcript Generation

## What is a Transcript?

A transcript is the textual representation of spoken audio.

Example:

**Audio**

```
Good morning everyone.
Let's begin today's sprint meeting.
Yesterday I completed the login module.
```

**Transcript**

```
Good morning everyone.
Let's begin today's sprint meeting.
Yesterday I completed the login module.
```

---

# Timestamp Generation

Each spoken sentence receives a start and end timestamp.

Example:

| Start | End | Text |
|--------|------|------|
| 0.00 | 3.42 | Good morning everyone. |
| 3.42 | 7.18 | Let's begin today's sprint meeting. |

These timestamps are useful for:

- Searching conversations
- Playback synchronization
- Speaker diarization
- Meeting navigation

---

# Transcript Builder

The Transcript Builder converts Whisper's raw output into a structured format used throughout the application.

Responsibilities include:

- Merge transcript segments.
- Preserve timestamps.
- Attach metadata.
- Create a standardized transcript object.
- Prepare data for database storage.

---

# Input

The Whisper module accepts audio generated by the preprocessing pipeline.

Supported input:

- WAV
- Mono audio
- PCM encoded
- Normalized audio
- Voice-only audio after VAD

Input source:

```
Audio Preprocessing Pipeline
```

---

# Output Format

Example:

```json
{
    "meeting_id":"meeting_001",
    "language":"en",
    "duration":144.81,
    "transcript":"Good morning everyone. Let's begin today's sprint meeting.",
    "segments":[
        {
            "segment_id":0,
            "start":0.0,
            "end":4.2,
            "text":"Good morning everyone."
        },
        {
            "segment_id":1,
            "start":4.2,
            "end":8.7,
            "text":"Let's begin today's sprint meeting."
        }
    ],
    "metadata":{
        "model":"small",
        "processing_time":2.13
    }
}
```

---

# Dependencies

The Whisper module depends on the following libraries.

| Package | Purpose |
|----------|---------|
| faster-whisper | Speech recognition |
| ctranslate2 | Optimized inference engine |
| ffmpeg | Audio decoding |
| numpy | Numerical processing |
| pydantic | Data validation |
| logging | Structured logging |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Configuration

Environment variables:

```env
WHISPER_MODEL=small
WHISPER_DEVICE=auto
```

Example options:

| Variable | Description |
|-----------|-------------|
| WHISPER_MODEL | Whisper model size |
| WHISPER_DEVICE | cpu, cuda or auto |

---

# Logging

Typical logs generated by the service:

```
INFO  Loading Faster Whisper model

INFO  Starting transcription

INFO  Language detected: English

INFO  Generated 48 transcript segments

INFO  Transcript generation completed

INFO  Processing completed successfully

ERROR  Audio file not found

ERROR  Model loading failed

ERROR  Transcription failed
```

---

# Error Handling

Custom exceptions include:

| Exception | Description |
|------------|-------------|
| AudioNotFoundError | Input audio missing |
| UnsupportedAudioError | Unsupported audio format |
| ModelLoadError | Whisper model failed to load |
| TranscriptionError | Error during transcription |

All exceptions are logged before being propagated.

---

# Testing

Run all Whisper unit tests:

```bash
pytest tests/
```

Expected tests include:

- Model loading
- Audio transcription
- Missing audio file
- Invalid audio
- Transcript generation
- Timestamp generation
- Transcript builder
- Exception handling

---
| Model     | Approx. Size | Speed (CPU) | Accuracy      | Best Use                         |
| --------- | -----------: | ----------- | ------------- | -------------------------------- |
| tiny      |       ~39 MB | ⭐⭐⭐⭐⭐| Basic         | Testing                          
| base      |       ~74 MB | ⭐⭐⭐⭐  | Good          | Small projects                   
| **small** |  **~244 MB** | ⭐⭐⭐     | **Very Good** | **our project** 
| medium    |      ~769 MB | ⭐⭐       | Higher        | Powerful CPUs/GPUs               
| large-v3  |      ~1.5 GB | ⭐          | Best          | High-end GPU servers             


# Performance

Approximate expectations on CPU:

| Model | Speed |
|--------|---------|
| tiny | Fastest |
| base | Fast |
| small | Recommended |
| medium | Moderate |
| large-v3 | Slow |

Performance depends on:

- CPU/GPU hardware
- Audio duration
- Model size
- Available system memory

---

# Future Integration

The output generated by this module becomes the input for future AI services.

```
Whisper Transcript
        │
        ▼
Speaker Diarization
        │
        ▼
LangGraph Workflow
        │
        ▼
Meeting Summarization
        │
        ▼
Action Item Extraction
        │
        ▼
Meeting Minutes Generation
```

These components are implemented in later phases and are outside the scope of this module.

---

# Best Practices

- Use normalized audio for best transcription accuracy.
- Prefer the `small` model during development.
- Use GPU acceleration for long meeting recordings.
- Ensure audio is preprocessed before transcription.
- Keep model configuration externalized through environment variables.
- Log all transcription events for debugging and monitoring.

---

# Conclusion

The Whisper ASR module forms the foundation of the AI pipeline by transforming meeting audio into structured textual transcripts. It bridges the gap between raw speech and intelligent 
language processing, enabling downstream services such as Speaker Diarization, LangGraph orchestration, Meeting Summarization, and Action Item Extraction. By leveraging Faster-Whisper, 
the service provides high transcription accuracy with efficient inference, making it suitable for scalable, production-ready meeting intelligence systems.
