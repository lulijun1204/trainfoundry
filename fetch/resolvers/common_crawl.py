"""Resolve Common Crawl requirements into direct HTTP files."""

import gzip
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fetch.models import CommonCrawlAcquisition, HttpAcquisition, HttpFileRequest


@dataclass(frozen=True)
class ResolvedCommonCrawl:
    acquisition: HttpAcquisition
    revision: str


class CommonCrawlResolver:
    """Look up a crawl and select one WET archive."""

    def resolve(self, request: CommonCrawlAcquisition) -> ResolvedCommonCrawl:
        collections = self._collections()
        collection = collections[0] if request.latest else collections[-1]
        crawl_id = collection["id"]
        paths = self._wet_paths(crawl_id)
        wet_path = paths[request.wet_file_index]
        url = f"https://data.commoncrawl.org/{wet_path}"
        return ResolvedCommonCrawl(
            acquisition=HttpAcquisition(
                files=(
                    HttpFileRequest(
                        url=url,
                        output_name=Path(wet_path).name,
                    ),
                )
            ),
            revision=crawl_id,
        )

    def identify(self, filename: str, search_limit: int = 3) -> tuple[str, str] | None:
        """Find the recent crawl that owns a previously downloaded WET file."""
        for collection in self._collections()[:search_limit]:
            crawl_id = collection["id"]
            for wet_path in self._wet_paths(crawl_id):
                if Path(wet_path).name == filename:
                    return crawl_id, f"https://data.commoncrawl.org/{wet_path}"
        return None

    @staticmethod
    def _collections() -> list[dict]:
        return json.load(
            urllib.request.urlopen(
                "https://index.commoncrawl.org/collinfo.json",
                timeout=30,
            )
        )

    @staticmethod
    def _wet_paths(crawl_id: str) -> list[str]:
        paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/wet.paths.gz"
        compressed = urllib.request.urlopen(paths_url, timeout=30).read()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as file:
            return [line.decode("utf-8").strip() for line in file if line.strip()]
