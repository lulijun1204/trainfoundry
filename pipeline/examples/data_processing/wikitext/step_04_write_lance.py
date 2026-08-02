"""Step 4: stream standardized Arrow batches into a new Lance dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.examples.data_processing.wikitext.processing import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_PATH,
    inspect_lance_dataset,
    write_lance_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="keep and inspect an existing Lance dataset instead of rebuilding it",
    )
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    if args.output.exists() and not overwrite:
        result = inspect_lance_dataset(args.output, limit=0)
        result["action"] = "skipped_existing"
        result["hint"] = "omit --no-overwrite to rebuild this exact output"
    else:
        result = write_lance_dataset(
            DEFAULT_DATASET_ROOT,
            args.output,
            batch_size=args.batch_size,
            overwrite=overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
