"""The sole executable entry point for the speech-service AI pipeline."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.ai_pipeline import AIPipeline
from preprocessing.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Parse CLI input, execute the AI pipeline, and print formatted JSON."""
    parser = argparse.ArgumentParser(description="Run meeting audio through the AI pipeline.")
    parser.add_argument("input_audio_path", type=Path, help="Original uploaded meeting audio")
    parser.add_argument("--meeting-id", default=None, help="Meeting identifier for the transcript")
    parser.add_argument("--content-type", default=None, help="Optional source MIME type")
    arguments = parser.parse_args()
    meeting_id = arguments.meeting_id or arguments.input_audio_path.stem
    started_at = perf_counter()
    try:
        result = AIPipeline().process(
            meeting_id=meeting_id,
            input_audio_path=arguments.input_audio_path,
            content_type=arguments.content_type,
        )
    except Exception as exc:
        logger.exception("Pipeline runner failed")
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 1

    output = result.to_dict()
    output["runner_processing_time"] = round(perf_counter() - started_at, 4)
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

