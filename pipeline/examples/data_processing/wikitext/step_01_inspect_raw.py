"""Step 1: view physical JSONL lines and their decoded values."""

from __future__ import annotations

import argparse
import json

from pipeline.examples.data_processing.wikitext.processing import (
    DEFAULT_DATASET_ROOT,
    iter_raw_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    print(f"dataset_root={DEFAULT_DATASET_ROOT}")
    for index, record in enumerate(iter_raw_records(DEFAULT_DATASET_ROOT)):
        if index >= args.limit:
            break
        print(
            json.dumps(
                {
                    "split": record.split,
                    "line_number": record.line_number,
                    "raw_bytes": repr(record.raw_bytes[:120]),
                    "decoded_value": record.value,
                    "parse_error": record.error_message,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
