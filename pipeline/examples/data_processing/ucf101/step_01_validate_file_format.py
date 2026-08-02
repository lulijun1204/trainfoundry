"""Step 1: validate UCF101 archives and video decodability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.examples.data_processing.ucf101.processing import (
    DEFAULT_DATASET_ROOT,
    validate_file_format,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-issue-samples", type=int, default=10)
    args = parser.parse_args()
    result = validate_file_format(
        args.dataset_root,
        full_decode=not args.probe_only,
        max_issue_samples=args.max_issue_samples,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
