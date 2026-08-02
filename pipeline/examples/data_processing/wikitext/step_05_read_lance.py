"""Step 5: reopen the standardized Lance dataset and inspect its rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.examples.data_processing.wikitext.processing import (
    DEFAULT_OUTPUT_PATH,
    inspect_lance_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    result = inspect_lance_dataset(args.input, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
