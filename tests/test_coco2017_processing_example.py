import json
import zipfile
from io import BytesIO

from PIL import Image

from pipeline.examples.data_processing.coco2017.processing import (
    validate_dataset,
    validate_file_format,
)


def _jpeg(width=10, height=8):
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def _payloads(*, width=10, caption="a red training image", bbox=None):
    image = {"id": 1, "file_name": "0001.jpg", "width": width, "height": 8}
    category = {"id": 1, "name": "person", "supercategory": "person"}
    return {
        "instances_val2017.json": {
            "images": [image],
            "categories": [category],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": bbox or [1, 1, 4, 4],
                    "area": 16,
                    "iscrowd": 0,
                    "segmentation": [[1, 1, 5, 1, 5, 5]],
                }
            ],
        },
        "captions_val2017.json": {
            "images": [image],
            "annotations": [{"id": 2, "image_id": 1, "caption": caption}],
        },
        "person_keypoints_val2017.json": {
            "images": [image],
            "categories": [category],
            "annotations": [
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1, 1, 4, 4],
                    "area": 16,
                    "iscrowd": 0,
                    "segmentation": [[1, 1, 5, 1, 5, 5]],
                    "keypoints": [2, 2, 2],
                    "num_keypoints": 1,
                }
            ],
        },
    }


def _write_fixture(root, payloads=None, image_bytes=None):
    root.mkdir()
    with zipfile.ZipFile(root / "val2017.zip", "w") as archive:
        archive.writestr("val2017/0001.jpg", image_bytes or _jpeg())
    with zipfile.ZipFile(root / "annotations_trainval2017.zip", "w") as archive:
        for name, payload in (payloads or _payloads()).items():
            archive.writestr(f"annotations/{name}", json.dumps(payload))


def test_coco_file_format_validation_fully_decodes_images(tmp_path):
    root = tmp_path / "coco2017"
    _write_fixture(root)

    summary = validate_file_format(root)

    assert summary.archive_count == 2
    assert summary.valid_archives == 2
    assert summary.image_files == 1
    assert summary.decoded_images == 1
    assert summary.annotation_files == 3
    assert summary.image_formats == {"JPEG": 1}
    assert summary.issue_counts == {}


def test_coco_file_format_validation_reports_broken_image(tmp_path):
    root = tmp_path / "coco2017"
    _write_fixture(root, image_bytes=b"not a jpeg")

    summary = validate_file_format(root)

    assert summary.decoded_images == 0
    assert summary.invalid_archives == 1
    assert summary.issue_counts == {"IMAGE_CORRUPT": 1}


def test_coco_data_validation_checks_references_and_geometry(tmp_path):
    root = tmp_path / "coco2017"
    _write_fixture(root)

    summary = validate_dataset(root)

    assert summary.image_count == 1
    assert summary.valid_images == 1
    assert summary.annotation_count == 3
    assert summary.valid_annotations == 3
    assert summary.annotation_type_counts == {
        "captions": 1,
        "instances": 1,
        "keypoints": 1,
    }
    assert summary.issue_counts == {}


def test_coco_data_validation_reports_metadata_and_annotation_errors(tmp_path):
    root = tmp_path / "coco2017"
    payloads = _payloads(width=11, caption=" ", bbox=[9, 1, 4, 4])
    payloads["person_keypoints_val2017.json"]["annotations"][0][
        "num_keypoints"
    ] = 0
    _write_fixture(root, payloads=payloads)

    summary = validate_dataset(root)

    assert summary.valid_images == 0
    assert summary.valid_annotations == 0
    assert summary.issue_counts == {
        "BBOX_OUT_OF_BOUNDS": 1,
        "CAPTION_INVALID": 1,
        "IMAGE_DIMENSION_MISMATCH": 1,
        "KEYPOINT_COUNT_MISMATCH": 1,
    }
