"""Step 3: show exact before/after text changes and the Arrow representation."""

from __future__ import annotations

import argparse
import json

from pipeline.examples.data_processing.wikitext.processing import (
    DEFAULT_DATASET_ROOT,
    iter_arrow_batches,
    iter_raw_records,
    normalize_text,
    validate_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    shown = 0
    for record in iter_raw_records(DEFAULT_DATASET_ROOT):
        if validate_record(record) is not None:
            continue
        before = record.value["text"]
        after = normalize_text(before)
        if before == after:
            continue
        print(
            json.dumps(
                {
                    "split": record.split,
                    "line_number": record.line_number,
                    "before": before,
                    "after": after,
                },
                ensure_ascii=False,
            )
        )
        shown += 1
        if shown >= args.limit:
            break

    first_batch = next(iter_arrow_batches(DEFAULT_DATASET_ROOT, batch_size=8))
    print("\nArrow schema:")
    print(first_batch.schema)
    print("\nFirst standardized Arrow rows:")
    print(json.dumps(first_batch.to_pylist(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
