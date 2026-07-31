from fetch.catalog import SOURCES
from fetch.models import (
    CommonCrawlAcquisition,
    HttpAcquisition,
    HuggingFaceAcquisition,
    MinariAcquisition,
)


def test_catalog_has_unique_requirements_for_all_supported_sources():
    assert len(SOURCES) == 7
    assert set(SOURCES) == {spec.source_id for spec in SOURCES.values()}


def test_catalog_separates_requirements_from_download_tools():
    assert isinstance(SOURCES["coco_2017_validation"].acquisition, HttpAcquisition)
    assert isinstance(SOURCES["ucf101"].acquisition, HttpAcquisition)
    assert isinstance(
        SOURCES["wikitext_2_raw"].acquisition,
        HuggingFaceAcquisition,
    )
    assert isinstance(
        SOURCES["common_crawl_wet"].acquisition,
        CommonCrawlAcquisition,
    )
    assert isinstance(
        SOURCES["d4rl_pointmaze_umaze_minari"].acquisition,
        MinariAcquisition,
    )
