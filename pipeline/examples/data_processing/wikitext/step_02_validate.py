"""Step 2: scan every row and explain why rows are accepted or rejected."""

from __future__ import annotations

import json

from pipeline.examples.data_processing.wikitext.processing import (
    DEFAULT_DATASET_ROOT,
    validate_dataset,
)


def main() -> None:
    summary = validate_dataset(DEFAULT_DATASET_ROOT)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
