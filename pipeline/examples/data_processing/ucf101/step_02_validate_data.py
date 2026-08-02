"""Step 2: validate UCF101 labels, folds, coverage, and leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.examples.data_processing.ucf101.processing import (
    DEFAULT_DATASET_ROOT,
    validate_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--skip-content-digest", action="store_true")
    parser.add_argument("--max-issue-samples", type=int, default=10)
    args = parser.parse_args()
    result = validate_dataset(
        args.dataset_root,
        check_content_duplicates=not args.skip_content_digest,
        max_issue_samples=args.max_issue_samples,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
