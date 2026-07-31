"""Materialize Hugging Face splits as portable JSONL files."""

from pathlib import Path
from typing import Any


class JsonlMaterializer:
    def materialize(self, dataset: Any, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for split, rows in dataset.items():
            output = destination / f"{split}.jsonl"
            rows.to_json(output, force_ascii=False)
            outputs.append(output)
        return outputs
