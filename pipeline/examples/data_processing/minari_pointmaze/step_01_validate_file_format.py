"""Step 1: validate the physical Minari package and every HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_DATASET_ROOT
from .file_validation import validate_file_format


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-issue-samples", type=int, default=10)
    args = parser.parse_args()
    result = validate_file_format(
        args.dataset_root, max_issue_samples=args.max_issue_samples
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
