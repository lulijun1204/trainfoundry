"""Declarative catalog of supported dataset sources."""

from fetch.models import (
    CommonCrawlAcquisition,
    DatasetMeta,
    HttpAcquisition,
    HttpFileRequest,
    HuggingFaceAcquisition,
    MinariAcquisition,
    OutputSpec,
)

SOURCES = {
    source.source_id: source
    for source in (
        DatasetMeta(
            source_id="wikitext_2_raw",
            modality="text",
            homepage="https://huggingface.co/datasets/Salesforce/wikitext",
            license="CC BY-SA 4.0",
            permitted_use="license_terms_apply",
            contains_pii="unknown",
            retention_policy="retain versioned raw JSONL",
            output=OutputSpec("paths.text_path", ("wikitext_2_raw",)),
            acquisition=HuggingFaceAcquisition(
                repo_id="Salesforce/wikitext",
                config="wikitext-2-raw-v1",
            ),
            purpose="PRETRAIN",
        ),
        DatasetMeta(
            source_id="dolly_15k",
            modality="text",
            homepage=(
                "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
            ),
            license="CC BY-SA 3.0",
            permitted_use="license_terms_apply",
            contains_pii="unknown",
            retention_policy="retain versioned raw JSONL",
            output=OutputSpec("paths.text_path", ("dolly_15k",)),
            acquisition=HuggingFaceAcquisition(
                repo_id="databricks/databricks-dolly-15k",
            ),
            purpose="SFT",
        ),
        DatasetMeta(
            source_id="hh_rlhf_helpful_base",
            modality="text",
            homepage="https://huggingface.co/datasets/Anthropic/hh-rlhf",
            license="MIT",
            permitted_use="license_terms_apply",
            contains_pii="unknown",
            retention_policy="retain versioned raw JSONL",
            output=OutputSpec("paths.text_path", ("hh_rlhf_helpful_base",)),
            acquisition=HuggingFaceAcquisition(
                repo_id="Anthropic/hh-rlhf",
                data_dir="helpful-base",
            ),
            purpose="RL",
        ),
        DatasetMeta(
            source_id="common_crawl_wet",
            modality="text",
            homepage="https://commoncrawl.org/get-started",
            license="Common Crawl terms; page-level rights may vary",
            permitted_use="license_and_source_terms_review",
            contains_pii="possible",
            retention_policy="honor source removals and deletion propagation",
            output=OutputSpec("paths.text_path", ("common_crawl",)),
            acquisition=CommonCrawlAcquisition(),
            purpose="PRETRAIN",
        ),
        DatasetMeta(
            source_id="coco_2017_validation",
            modality="image",
            homepage="https://cocodataset.org/",
            license="image licenses vary; annotations CC BY 4.0",
            permitted_use="license_needs_review",
            contains_pii="possible",
            retention_policy="retain raw archives; honor source removals",
            output=OutputSpec(
                "paths.multimodal_path",
                ("image", "coco2017"),
            ),
            acquisition=HttpAcquisition(
                files=(
                    HttpFileRequest(
                        url="http://images.cocodataset.org/zips/val2017.zip",
                        output_name="val2017.zip",
                        expected_bytes=815_585_330,
                    ),
                    HttpFileRequest(
                        url=(
                            "http://images.cocodataset.org/annotations/"
                            "annotations_trainval2017.zip"
                        ),
                        output_name="annotations_trainval2017.zip",
                        expected_bytes=252_907_541,
                    ),
                ),
                revision="2017",
            ),
            purpose="BENCHMARK",
        ),
        DatasetMeta(
            source_id="ucf101",
            modality="video",
            homepage="https://www.crcv.ucf.edu/research/data-sets/ucf101/",
            license="license_needs_review",
            permitted_use="local_research_only",
            contains_pii="possible",
            retention_policy="local research copy; review source terms",
            output=OutputSpec(
                "paths.multimodal_path",
                ("video", "ucf101"),
            ),
            acquisition=HttpAcquisition(
                files=(
                    HttpFileRequest(
                        url="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
                        output_name="UCF101.rar",
                        expected_bytes=6_932_971_618,
                    ),
                    HttpFileRequest(
                        url=(
                            "https://www.crcv.ucf.edu/data/UCF101/"
                            "UCF101TrainTestSplits-RecognitionTask.zip"
                        ),
                        output_name="UCF101TrainTestSplits.zip",
                        expected_bytes=113_943,
                    ),
                ),
                revision="UCF101",
            ),
            purpose="BENCHMARK",
        ),
        DatasetMeta(
            source_id="d4rl_pointmaze_umaze_minari",
            modality="robot",
            homepage="https://minari.farama.org/datasets/pointmaze/umaze/",
            license="dataset metadata/source terms apply",
            permitted_use="research",
            contains_pii=False,
            retention_policy="retain versioned Minari dataset",
            output=OutputSpec("paths.robot_path"),
            acquisition=MinariAcquisition("D4RL/pointmaze/umaze-v2"),
            purpose="RL",
        ),
    )
}

HUGGING_FACE_SOURCE_IDS = (
    "wikitext_2_raw",
    "dolly_15k",
    "hh_rlhf_helpful_base",
)

REMAINING_SOURCE_IDS = (
    "coco_2017_validation",
    "ucf101",
    "d4rl_pointmaze_umaze_minari",
)


SOURCE_GROUPS = {
    "huggingface": HUGGING_FACE_SOURCE_IDS,
    "non_text": REMAINING_SOURCE_IDS,
}


def get_source(source_id: str) -> DatasetMeta:
    """Return a source definition or fail with a useful error."""
    try:
        return SOURCES[source_id]
    except KeyError as exc:
        supported = ", ".join(sorted(SOURCES))
        raise KeyError(f"Unknown source {source_id!r}; supported: {supported}") from exc
