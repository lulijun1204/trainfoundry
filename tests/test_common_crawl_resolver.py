from fetch.models import CommonCrawlAcquisition, HttpAcquisition
from fetch.resolvers.common_crawl import CommonCrawlResolver


def test_resolver_turns_dynamic_requirement_into_http_request(monkeypatch):
    resolver = CommonCrawlResolver()
    monkeypatch.setattr(
        resolver,
        "_collections",
        lambda: [{"id": "CC-MAIN-TEST"}],
    )
    monkeypatch.setattr(
        resolver,
        "_wet_paths",
        lambda crawl_id: ["crawl-data/CC-MAIN-TEST/segments/one/example.warc.wet.gz"],
    )

    resolved = resolver.resolve(CommonCrawlAcquisition())

    assert resolved.revision == "CC-MAIN-TEST"
    assert isinstance(resolved.acquisition, HttpAcquisition)
    assert resolved.acquisition.files[0].output_name == "example.warc.wet.gz"
