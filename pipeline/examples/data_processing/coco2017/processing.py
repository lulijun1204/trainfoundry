"""Concrete COCO 2017 file-format and annotation validation.

The example intentionally keeps physical media validation separate from
logical sample validation. It does not clean, transform, or materialize data.
"""

from __future__ import annotations

import json
import math
import stat
import unicodedata
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "model_data/multimodal/image/coco2017"
IMAGES_ARCHIVE = "val2017.zip"
ANNOTATIONS_ARCHIVE = "annotations_trainval2017.zip"
EXPECTED_ARCHIVES = (IMAGES_ARCHIVE, ANNOTATIONS_ARCHIVE)
EXPECTED_VAL_ANNOTATIONS = (
    "instances_val2017.json",
    "captions_val2017.json",
    "person_keypoints_val2017.json",
)
_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    source_file: str
    location: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class FileFormatSummary:
    archive_count: int
    valid_archives: int
    invalid_archives: int
    total_compressed_bytes: int
    image_files: int
    decoded_images: int
    annotation_files: int
    image_formats: dict[str, int]
    image_modes: dict[str, int]
    min_width: int | None
    max_width: int | None
    min_height: int | None
    max_height: int | None
    archive_sha256: dict[str, str]
    issue_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DataValidationSummary:
    image_count: int
    valid_images: int
    rejected_images: int
    annotation_count: int
    valid_annotations: int
    rejected_annotations: int
    category_count: int
    annotation_type_counts: dict[str, int]
    issue_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Collector:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_issue_samples must be positive")
        self.limit = limit
        self.counts: Counter[str] = Counter()
        self.samples: list[ValidationIssue] = []

    def add(self, source: str, location: str, code: str, message: str) -> None:
        self.counts[code] += 1
        sampled_codes = {sample.code for sample in self.samples}
        if len(self.samples) < self.limit and code not in sampled_codes:
            self.samples.append(ValidationIssue(source, location, code, message))


def validate_file_format(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> FileFormatSummary:
    """Validate ZIP containers and fully decode every COCO validation image."""
    # Step 01 校验物理载体。它不理解 bbox 等任务语义，也不会改写原始 ZIP。
    issues = _Collector(max_issue_samples)
    invalid_archives: set[str] = set()
    archive_sha256: dict[str, str] = {}
    total_bytes = 0
    image_count = 0
    decoded_images = 0
    annotation_files = 0
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []

    paths = {name: dataset_root / name for name in EXPECTED_ARCHIVES}
    for name, path in paths.items():
        if not path.is_file():
            issues.add(name, name, "MISSING_FILE", "required archive is missing")
            invalid_archives.add(name)
            continue
        total_bytes += path.stat().st_size
        # 摘要用于来源审计；magic bytes 用来防止只改后缀的伪 ZIP。
        archive_sha256[name], magic = _digest_and_magic(path)
        if magic not in {b"PK\x03\x04", b"PK\x05\x06"}:
            issues.add(name, name, "INVALID_ZIP_MAGIC", f"unexpected magic: {magic!r}")
            invalid_archives.add(name)
            continue
        before = sum(issues.counts.values())
        try:
            with zipfile.ZipFile(path) as archive:
                _validate_zip_members(archive, name, issues)
                corrupt = archive.testzip()
                if corrupt is not None:
                    issues.add(name, corrupt, "ZIP_CRC_ERROR", "member CRC check failed")
                if name == IMAGES_ARCHIVE:
                    # 图片必须逐张完整解码。只读尺寸/文件头无法发现尾部截断。
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        image_count += 1
                        observation = _decode_image_member(archive, info, issues)
                        if observation is None:
                            continue
                        decoded_images += 1
                        image_format, mode, width, height = observation
                        formats[image_format] += 1
                        modes[mode] += 1
                        widths.append(width)
                        heights.append(height)
                else:
                    # 文件级只验证 UTF-8/JSON 能否解析，引用关系由 Step 02 处理。
                    for info in archive.infolist():
                        if info.is_dir() or not info.filename.lower().endswith(".json"):
                            continue
                        annotation_files += 1
                        try:
                            with archive.open(info) as stream:
                                json.loads(stream.read().decode("utf-8"))
                        except UnicodeDecodeError as exc:
                            issues.add(name, info.filename, "INVALID_UTF8", str(exc))
                        except json.JSONDecodeError as exc:
                            issues.add(name, info.filename, "INVALID_JSON", str(exc))
        except (OSError, zipfile.BadZipFile, EOFError) as exc:
            issues.add(name, name, "ZIP_CORRUPT", str(exc))
        if sum(issues.counts.values()) > before:
            invalid_archives.add(name)

    if image_count == 0:
        issues.add(IMAGES_ARCHIVE, IMAGES_ARCHIVE, "NO_IMAGES", "archive has no images")
        invalid_archives.add(IMAGES_ARCHIVE)

    present = sum(path.is_file() for path in paths.values())
    return FileFormatSummary(
        archive_count=present,
        valid_archives=max(0, present - len(invalid_archives & set(paths))),
        invalid_archives=len(invalid_archives),
        total_compressed_bytes=total_bytes,
        image_files=image_count,
        decoded_images=decoded_images,
        annotation_files=annotation_files,
        image_formats=dict(sorted(formats.items())),
        image_modes=dict(sorted(modes.items())),
        min_width=min(widths) if widths else None,
        max_width=max(widths) if widths else None,
        min_height=min(heights) if heights else None,
        max_height=max(heights) if heights else None,
        archive_sha256=dict(sorted(archive_sha256.items())),
        issue_counts=dict(sorted(issues.counts.items())),
        issue_samples=tuple(issues.samples),
    )


def validate_dataset(
    dataset_root: Path,
    *,
    max_issue_samples: int = 10,
) -> DataValidationSummary:
    """Validate COCO image metadata, references, and task annotations."""
    # Step 02 将 instances 作为 image/category 的基准，再与 captions、keypoints 对齐。
    issues = _Collector(max_issue_samples)
    images_path = dataset_root / IMAGES_ARCHIVE
    annotations_path = dataset_root / ANNOTATIONS_ARCHIVE
    if not images_path.is_file() or not annotations_path.is_file():
        for path in (images_path, annotations_path):
            if not path.is_file():
                issues.add(path.name, path.name, "MISSING_FILE", "required archive is missing")
        return _data_summary(0, set(), 0, set(), 0, Counter(), issues)

    try:
        image_members, actual_dimensions = _read_image_metadata(images_path, issues)
        payloads = _read_val_annotations(annotations_path, issues)
    except (OSError, zipfile.BadZipFile) as exc:
        issues.add(str(dataset_root), str(dataset_root), "CONTAINER_CORRUPT", str(exc))
        return _data_summary(0, set(), 0, set(), 0, Counter(), issues)

    instances = payloads.get("instances_val2017.json")
    if not isinstance(instances, dict):
        return _data_summary(0, set(), 0, set(), 0, Counter(), issues)

    rejected_images: set[str] = set()
    rejected_annotations: set[str] = set()
    image_by_id, category_ids = _validate_canonical_images(
        instances,
        image_members,
        actual_dimensions,
        rejected_images,
        issues,
    )
    annotation_types: Counter[str] = Counter()
    annotation_total = 0

    for filename, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        _validate_image_array_consistency(
            filename,
            payload,
            image_by_id,
            rejected_images,
            issues,
        )
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            issues.add(filename, "annotations", "SCHEMA_ERROR", "annotations must be an array")
            continue
        # 三种标注共享 image 引用，但字段契约不同，先按文件名确定任务类型。
        kind = _annotation_kind(filename)
        annotation_types[kind] += len(annotations)
        annotation_total += len(annotations)
        _validate_annotations(
            filename,
            kind,
            annotations,
            image_by_id,
            category_ids,
            rejected_annotations,
            issues,
        )

    return _data_summary(
        len(image_by_id),
        rejected_images,
        annotation_total,
        rejected_annotations,
        len(category_ids),
        annotation_types,
        issues,
    )


def _validate_zip_members(
    archive: zipfile.ZipFile,
    source: str,
    issues: _Collector,
) -> None:
    # 先检查成员表而不是直接解包，避免路径穿越、链接和压缩炸弹风险。
    seen: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if name in seen:
            issues.add(source, name, "DUPLICATE_MEMBER", "duplicate ZIP member")
        seen.add(name)
        normalized = PurePosixPath(name.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            issues.add(source, name, "UNSAFE_MEMBER", "absolute or parent path in ZIP")
        if info.flag_bits & 0x1:
            issues.add(source, name, "ENCRYPTED_MEMBER", "encrypted ZIP member")
        if stat.S_ISLNK(info.external_attr >> 16):
            issues.add(source, name, "SYMLINK_MEMBER", "symbolic link in ZIP")
        ratio = info.file_size / max(1, info.compress_size)
        if ratio > 1_000:
            issues.add(source, name, "SUSPICIOUS_COMPRESSION", f"compression ratio {ratio:.1f}")


def _decode_image_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    issues: _Collector,
) -> tuple[str, str, int, int] | None:
    source = archive.filename and Path(archive.filename).name or IMAGES_ARCHIVE
    if not info.filename.lower().endswith((".jpg", ".jpeg")):
        issues.add(source, info.filename, "UNEXPECTED_IMAGE_SUFFIX", "COCO val image must be JPEG")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # verify() 检查编码结构，但会使 Image 对象不可继续使用。
            with archive.open(info) as stream, Image.open(stream) as image:
                image_format = image.format or "UNKNOWN"
                image.verify()
            # 因此必须重新打开并 load()，强制读取全部像素数据。
            with archive.open(info) as stream, Image.open(stream) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                orientation = image.getexif().get(274)
        if image_format != "JPEG":
            issues.add(source, info.filename, "FORMAT_MISMATCH", f"decoded as {image_format}")
        if width < 1 or height < 1:
            issues.add(source, info.filename, "IMAGE_DIMENSIONS", f"invalid size {width}x{height}")
        if orientation is not None and orientation not in range(1, 9):
            issues.add(source, info.filename, "EXIF_ORIENTATION", f"invalid orientation {orientation}")
        return image_format, mode, width, height
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        issues.add(source, info.filename, "IMAGE_CORRUPT", str(exc))
        return None


def _read_image_metadata(
    path: Path,
    issues: _Collector,
) -> tuple[set[str], dict[str, tuple[int, int]]]:
    members: set[str] = set()
    dimensions: dict[str, tuple[int, int]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            filename = Path(info.filename).name
            if filename in members:
                issues.add(path.name, info.filename, "DUPLICATE_IMAGE_NAME", filename)
            members.add(filename)
            try:
                with archive.open(info) as stream, Image.open(stream) as image:
                    dimensions[filename] = image.size
            except (OSError, UnidentifiedImageError) as exc:
                issues.add(path.name, info.filename, "IMAGE_CORRUPT", str(exc))
    return members, dimensions


def _read_val_annotations(path: Path, issues: _Collector) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        by_name = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        for filename in EXPECTED_VAL_ANNOTATIONS:
            info = by_name.get(filename)
            if info is None:
                issues.add(path.name, filename, "MISSING_ANNOTATION_FILE", "required validation annotation is missing")
                continue
            try:
                with archive.open(info) as stream:
                    payloads[filename] = json.loads(stream.read().decode("utf-8"))
            except UnicodeDecodeError as exc:
                issues.add(path.name, filename, "INVALID_UTF8", str(exc))
            except json.JSONDecodeError as exc:
                issues.add(path.name, filename, "INVALID_JSON", str(exc))
    return payloads


def _validate_canonical_images(
    payload: dict[str, Any],
    image_members: set[str],
    actual_dimensions: dict[str, tuple[int, int]],
    rejected: set[str],
    issues: _Collector,
) -> tuple[dict[Any, dict[str, Any]], set[Any]]:
    # instances 的 images/categories 是本示例的规范索引，其他标注文件向它对齐。
    images = payload.get("images")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(categories, list):
        issues.add("instances_val2017.json", "root", "SCHEMA_ERROR", "images and categories must be arrays")
        return {}, set()
    image_by_id: dict[Any, dict[str, Any]] = {}
    file_names: set[str] = set()
    for index, image in enumerate(images):
        location = f"images[{index}]"
        if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
            rejected.add(location)
            issues.add("instances_val2017.json", location, "IMAGE_METADATA_INVALID", "id and file_name are required")
            continue
        image_id = image["id"]
        filename = image["file_name"]
        key = str(image_id)
        if image_id in image_by_id:
            rejected.add(key)
            issues.add("instances_val2017.json", location, "DUPLICATE_IMAGE_ID", str(image_id))
        if not isinstance(filename, str) or filename in file_names:
            rejected.add(key)
            issues.add("instances_val2017.json", location, "DUPLICATE_IMAGE_NAME", str(filename))
        file_names.add(filename)
        image_by_id[image_id] = image
        if filename not in image_members:
            rejected.add(key)
            issues.add("instances_val2017.json", location, "IMAGE_FILE_MISSING", str(filename))
            continue
        expected = (image.get("width"), image.get("height"))
        actual = actual_dimensions.get(filename)
        if actual is None or expected != actual:
            rejected.add(key)
            issues.add("instances_val2017.json", location, "IMAGE_DIMENSION_MISMATCH", f"metadata={expected}, decoded={actual}")
    for orphan in sorted(image_members - file_names):
        # ZIP 中存在、JSON 中不存在的图片通常意味着打包或版本混用。
        rejected.add(f"orphan:{orphan}")
        issues.add(IMAGES_ARCHIVE, orphan, "ORPHAN_IMAGE", "image is absent from instances metadata")

    category_ids: set[Any] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or "id" not in category or "name" not in category:
            issues.add("instances_val2017.json", f"categories[{index}]", "CATEGORY_INVALID", "id and name are required")
            continue
        if category["id"] in category_ids:
            issues.add("instances_val2017.json", f"categories[{index}]", "DUPLICATE_CATEGORY_ID", str(category["id"]))
        category_ids.add(category["id"])
    return image_by_id, category_ids


def _validate_image_array_consistency(
    filename: str,
    payload: dict[str, Any],
    canonical: dict[Any, dict[str, Any]],
    rejected: set[str],
    issues: _Collector,
) -> None:
    images = payload.get("images")
    if not isinstance(images, list):
        issues.add(filename, "images", "SCHEMA_ERROR", "images must be an array")
        return
    for index, image in enumerate(images):
        if not isinstance(image, dict) or "id" not in image:
            issues.add(filename, f"images[{index}]", "IMAGE_METADATA_INVALID", "image id is required")
            continue
        expected = canonical.get(image["id"])
        if expected is None:
            rejected.add(str(image["id"]))
            issues.add(filename, f"images[{index}]", "UNKNOWN_IMAGE_ID", str(image["id"]))
            continue
        for field in ("file_name", "width", "height"):
            if image.get(field) != expected.get(field):
                rejected.add(str(image["id"]))
                issues.add(filename, f"images[{index}]", "IMAGE_METADATA_MISMATCH", f"field {field}")
                break


def _validate_annotations(
    filename: str,
    kind: str,
    annotations: list[Any],
    images: dict[Any, dict[str, Any]],
    category_ids: set[Any],
    rejected: set[str],
    issues: _Collector,
) -> None:
    # annotation 的主键和外键先校验，再执行各任务自己的内容约束。
    seen_ids: set[Any] = set()
    for index, annotation in enumerate(annotations):
        key = f"{filename}:{index}"
        if not isinstance(annotation, dict):
            rejected.add(key)
            issues.add(filename, f"annotations[{index}]", "ANNOTATION_INVALID", "annotation must be an object")
            continue
        annotation_id = annotation.get("id")
        key = f"{filename}:{annotation_id if annotation_id is not None else index}"
        if annotation_id is None or annotation_id in seen_ids:
            rejected.add(key)
            issues.add(filename, f"annotations[{index}]", "DUPLICATE_ANNOTATION_ID", str(annotation_id))
        seen_ids.add(annotation_id)
        image = images.get(annotation.get("image_id"))
        if image is None:
            rejected.add(key)
            issues.add(filename, f"annotations[{index}]", "UNKNOWN_IMAGE_ID", str(annotation.get("image_id")))
            continue
        if kind in {"instances", "keypoints"} and annotation.get("category_id") not in category_ids:
            rejected.add(key)
            issues.add(filename, f"annotations[{index}]", "UNKNOWN_CATEGORY_ID", str(annotation.get("category_id")))
        before = sum(issues.counts.values())
        if kind == "captions":
            _validate_caption(filename, index, annotation, issues)
        else:
            _validate_geometry(filename, index, annotation, image, kind, issues)
        if sum(issues.counts.values()) > before:
            rejected.add(key)


def _validate_caption(source: str, index: int, annotation: dict[str, Any], issues: _Collector) -> None:
    caption = annotation.get("caption")
    location = f"annotations[{index}]"
    if not isinstance(caption, str) or not caption.strip():
        issues.add(source, location, "CAPTION_INVALID", "caption must be a non-empty string")
        return
    controls = sorted({ord(char) for char in caption if unicodedata.category(char) == "Cc" and char not in "\n\r\t"})
    if controls:
        issues.add(source, location, "CAPTION_CONTROL_CHARACTER", ", ".join(f"U+{value:04X}" for value in controls))


def _validate_geometry(
    source: str,
    index: int,
    annotation: dict[str, Any],
    image: dict[str, Any],
    kind: str,
    issues: _Collector,
) -> None:
    location = f"annotations[{index}]"
    width = image.get("width", 0)
    height = image.get("height", 0)
    bbox = annotation.get("bbox")
    if not _numeric_list(bbox, 4):
        issues.add(source, location, "BBOX_INVALID", "bbox must contain four finite numbers")
    else:
        x, y, box_width, box_height = bbox
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > width + 1 or y + box_height > height + 1:
            issues.add(source, location, "BBOX_OUT_OF_BOUNDS", f"bbox={bbox}, image={width}x{height}")
    area = annotation.get("area")
    if not _number(area) or area < 0:
        issues.add(source, location, "AREA_INVALID", f"area={area!r}")
    _validate_segmentation(source, location, annotation.get("segmentation"), width, height, issues)
    if kind == "keypoints":
        _validate_keypoints(source, location, annotation, width, height, issues)


def _validate_segmentation(source: str, location: str, value: Any, width: int, height: int, issues: _Collector) -> None:
    # COCO segmentation 有两种合法表示：普通目标使用 polygon，crowd 常用 RLE。
    if isinstance(value, list):
        for polygon in value:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 or not all(_number(item) for item in polygon):
                issues.add(source, location, "SEGMENTATION_INVALID", "polygon must contain at least three finite coordinate pairs")
                return
            for x, y in zip(polygon[0::2], polygon[1::2], strict=True):
                if x < -1 or y < -1 or x > width + 1 or y > height + 1:
                    issues.add(source, location, "SEGMENTATION_OUT_OF_BOUNDS", f"point=({x}, {y})")
                    return
    elif isinstance(value, dict):
        size = value.get("size")
        counts = value.get("counts")
        if not _numeric_list(size, 2) or tuple(size) != (height, width) or not isinstance(counts, (str, list)):
            issues.add(source, location, "RLE_INVALID", "RLE requires matching [height, width] and counts")
    else:
        issues.add(source, location, "SEGMENTATION_INVALID", "segmentation must be polygon or RLE")


def _validate_keypoints(source: str, location: str, annotation: dict[str, Any], width: int, height: int, issues: _Collector) -> None:
    keypoints = annotation.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) % 3 or not all(_number(item) for item in keypoints):
        issues.add(source, location, "KEYPOINTS_INVALID", "keypoints must be finite x,y,v triples")
        return
    visible = 0
    for x, y, visibility in zip(keypoints[0::3], keypoints[1::3], keypoints[2::3], strict=True):
        if visibility not in {0, 1, 2}:
            issues.add(source, location, "KEYPOINT_VISIBILITY", str(visibility))
            return
        if visibility > 0:
            visible += 1
            if x < -1 or y < -1 or x > width + 1 or y > height + 1:
                issues.add(source, location, "KEYPOINT_OUT_OF_BOUNDS", f"point=({x}, {y})")
                return
    if annotation.get("num_keypoints") != visible:
        issues.add(source, location, "KEYPOINT_COUNT_MISMATCH", f"declared={annotation.get('num_keypoints')}, actual={visible}")


def _annotation_kind(filename: str) -> str:
    if filename.startswith("captions_"):
        return "captions"
    if filename.startswith("person_keypoints_"):
        return "keypoints"
    return "instances"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_list(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_number(item) for item in value)


def _data_summary(
    image_count: int,
    rejected_images: set[str],
    annotation_count: int,
    rejected_annotations: set[str],
    category_count: int,
    annotation_types: Counter[str],
    issues: _Collector,
) -> DataValidationSummary:
    return DataValidationSummary(
        image_count=image_count,
        valid_images=max(0, image_count - len(rejected_images)),
        rejected_images=len(rejected_images),
        annotation_count=annotation_count,
        valid_annotations=max(0, annotation_count - len(rejected_annotations)),
        rejected_annotations=len(rejected_annotations),
        category_count=category_count,
        annotation_type_counts=dict(sorted(annotation_types.items())),
        issue_counts=dict(sorted(issues.counts.items())),
        issue_samples=tuple(issues.samples),
    )


def _digest_and_magic(path: Path) -> tuple[str, bytes]:
    digest = sha256()
    magic = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            if not magic:
                magic = chunk[:4]
            digest.update(chunk)
    return digest.hexdigest(), magic
