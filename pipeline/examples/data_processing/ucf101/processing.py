"""Concrete UCF101 archive, video-stream, and split validation."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "model_data/multimodal/video/ucf101"
VIDEO_ARCHIVE = "UCF101.rar"
SPLIT_ARCHIVE = "UCF101TrainTestSplits.zip"
EXPECTED_VIDEO_COUNT = 13_320
EXPECTED_CLASS_COUNT = 101
EXPECTED_FOLDS = (1, 2, 3)
_RAR_MAGICS = (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")
_VIDEO_PATTERN = re.compile(r"^v_(?P<class>.+)_g(?P<group>\d+)_c(?P<clip>\d+)\.avi$")
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
    video_files: int
    probed_videos: int
    fully_decoded_videos: int
    container_formats: dict[str, int]
    video_codecs: dict[str, int]
    min_width: int | None
    max_width: int | None
    min_height: int | None
    max_height: int | None
    min_duration_seconds: float | None
    max_duration_seconds: float | None
    archive_sha256: dict[str, str]
    issue_counts: dict[str, int]
    issue_samples: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DataValidationSummary:
    class_count: int
    video_count: int
    fold_count: int
    split_entries: int
    content_digests_checked: bool
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
        if len(self.samples) < self.limit and code not in {item.code for item in self.samples}:
            self.samples.append(ValidationIssue(source, location, code, message))


def validate_file_format(
    dataset_root: Path,
    *,
    full_decode: bool = True,
    expected_video_count: int | None = EXPECTED_VIDEO_COUNT,
    max_issue_samples: int = 10,
) -> FileFormatSummary:
    """Validate both archives, probe every video, and normally decode every frame."""
    # 文件级证据逐层增强：归档可解包 -> 媒体流可探测 -> 每一帧可解码。
    # 三层不能互相替代，例如 ffprobe 成功仍可能存在中后段坏帧。
    issues = _Collector(max_issue_samples)
    rar_path = dataset_root / VIDEO_ARCHIVE
    split_path = dataset_root / SPLIT_ARCHIVE
    digests: dict[str, str] = {}
    invalid: set[str] = set()

    for path in (rar_path, split_path):
        if not path.is_file():
            issues.add(path.name, path.name, "MISSING_FILE", "required archive is missing")
            invalid.add(path.name)
        else:
            digests[path.name] = _digest(path)

    if split_path.is_file():
        before = sum(issues.counts.values())
        _validate_split_zip_container(split_path, issues)
        if sum(issues.counts.values()) > before:
            invalid.add(split_path.name)

    members: list[str] = []
    if rar_path.is_file():
        before = sum(issues.counts.values())
        if not _has_rar_magic(rar_path):
            issues.add(rar_path.name, rar_path.name, "INVALID_RAR_MAGIC", "not a RAR4/RAR5 archive")
        else:
            members = _list_rar(rar_path, issues)
            _validate_archive_members(rar_path.name, members, issues)
        if sum(issues.counts.values()) > before:
            invalid.add(rar_path.name)

    # 先从 RAR 成员表建立期望集合，解包后再逐一对账，防止静默漏文件。
    video_members = [name for name in members if name.lower().endswith(".avi")]
    if expected_video_count is not None and len(video_members) != expected_video_count:
        issues.add(VIDEO_ARCHIVE, "videos", "VIDEO_COUNT_MISMATCH", f"expected {expected_video_count}, got {len(video_members)}")
        invalid.add(VIDEO_ARCHIVE)

    probed = decoded = 0
    formats: Counter[str] = Counter()
    codecs: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    durations: list[float] = []
    if video_members and rar_path.is_file():
        # 临时展开只服务本次运行，退出作用域后自动清理，不产生 DatasetVersion。
        with tempfile.TemporaryDirectory(prefix="trainfoundry-ucf101-") as temporary:
            extracted = _extract_rar(rar_path, Path(temporary), issues)
            by_name = {_canonical_video(path): path for path in extracted}
            for member in video_members:
                canonical = _canonical_video(member)
                path = by_name.get(canonical)
                if path is None:
                    issues.add(VIDEO_ARCHIVE, member, "EXTRACTION_MISSING", "listed video was not extracted")
                    continue
                # probe 负责容器/流元数据，decode 负责完整帧数据，两类结果分开计数。
                observation = _probe_video(path, issues, canonical)
                if observation is None:
                    continue
                probed += 1
                container, codec, width, height, duration = observation
                formats[container] += 1
                codecs[codec] += 1
                widths.append(width)
                heights.append(height)
                durations.append(duration)
                if full_decode and _decode_video(path, issues, canonical):
                    decoded += 1
        if issues.counts:
            invalid.add(VIDEO_ARCHIVE)

    present = sum(path.is_file() for path in (rar_path, split_path))
    return FileFormatSummary(
        archive_count=present,
        valid_archives=max(0, present - len(invalid)),
        video_files=len(video_members),
        probed_videos=probed,
        fully_decoded_videos=decoded,
        container_formats=dict(sorted(formats.items())),
        video_codecs=dict(sorted(codecs.items())),
        min_width=min(widths) if widths else None,
        max_width=max(widths) if widths else None,
        min_height=min(heights) if heights else None,
        max_height=max(heights) if heights else None,
        min_duration_seconds=min(durations) if durations else None,
        max_duration_seconds=max(durations) if durations else None,
        archive_sha256=dict(sorted(digests.items())),
        issue_counts=dict(sorted(issues.counts.items())),
        issue_samples=tuple(issues.samples),
    )


def validate_dataset(
    dataset_root: Path,
    *,
    strict_official_counts: bool = True,
    check_content_duplicates: bool = True,
    max_issue_samples: int = 10,
) -> DataValidationSummary:
    """Validate class labels, three official folds, coverage, and leakage."""
    # 数据级不判断视频画质，而是验证“样本、标签、划分”之间的逻辑关系。
    issues = _Collector(max_issue_samples)
    rar_path = dataset_root / VIDEO_ARCHIVE
    split_path = dataset_root / SPLIT_ARCHIVE
    if not rar_path.is_file() or not split_path.is_file():
        for path in (rar_path, split_path):
            if not path.is_file():
                issues.add(path.name, path.name, "MISSING_FILE", "required archive is missing")
        return _data_summary(0, 0, 0, 0, False, issues)

    members = _list_rar(rar_path, issues)
    videos = {_canonical_video(name) for name in members if name.lower().endswith(".avi")}
    try:
        with zipfile.ZipFile(split_path) as archive:
            files = {
                Path(info.filename).name: archive.read(info).decode("utf-8-sig")
                for info in archive.infolist()
                if not info.is_dir()
            }
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        issues.add(SPLIT_ARCHIVE, SPLIT_ARCHIVE, "SPLIT_ARCHIVE_INVALID", str(exc))
        return _data_summary(0, len(videos), 0, 0, False, issues)

    classes = _parse_classes(files.get("classInd.txt"), issues)
    if strict_official_counts:
        if len(classes) != EXPECTED_CLASS_COUNT:
            issues.add(SPLIT_ARCHIVE, "classInd.txt", "CLASS_COUNT_MISMATCH", f"expected 101, got {len(classes)}")
        if len(videos) != EXPECTED_VIDEO_COUNT:
            issues.add(VIDEO_ARCHIVE, "videos", "VIDEO_COUNT_MISMATCH", f"expected 13320, got {len(videos)}")

    folds: dict[int, tuple[dict[str, int], set[str]]] = {}
    split_entries = 0
    for fold in EXPECTED_FOLDS:
        train = _parse_train(files.get(f"trainlist0{fold}.txt"), fold, issues)
        test = _parse_test(files.get(f"testlist0{fold}.txt"), fold, issues)
        folds[fold] = (train, test)
        split_entries += len(train) + len(test)
        # 每一折独立检查引用、覆盖、标签和 group 泄漏。
        _validate_fold(fold, train, test, videos, classes, issues)

    digests_checked = False
    if check_content_duplicates and videos:
        # 文件名不同不代表内容不同；字节摘要可补充检测精确内容泄漏。
        with tempfile.TemporaryDirectory(prefix="trainfoundry-ucf101-hash-") as temporary:
            extracted = _extract_rar(rar_path, Path(temporary), issues)
            hashes = {_canonical_video(path): _digest(path) for path in extracted}
            _validate_content_leakage(folds, hashes, issues)
            digests_checked = True

    return _data_summary(len(classes), len(videos), len(folds), split_entries, digests_checked, issues)


def _validate_split_zip_container(path: Path, issues: _Collector) -> None:
    try:
        with path.open("rb") as stream:
            if stream.read(4) not in {b"PK\x03\x04", b"PK\x05\x06"}:
                issues.add(path.name, path.name, "INVALID_ZIP_MAGIC", "split archive is not ZIP")
                return
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                issues.add(path.name, corrupt, "ZIP_CRC_ERROR", "member CRC check failed")
            for info in archive.infolist():
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if member.is_absolute() or ".." in member.parts:
                    issues.add(path.name, info.filename, "UNSAFE_MEMBER", "absolute or parent path in ZIP")
                if stat.S_ISLNK(info.external_attr >> 16):
                    issues.add(path.name, info.filename, "SYMLINK_MEMBER", "symbolic link in ZIP")
                if not info.is_dir():
                    archive.read(info).decode("utf-8-sig")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        issues.add(path.name, path.name, "ZIP_CORRUPT", str(exc))


def _has_rar_magic(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(8)
    return any(header.startswith(magic) for magic in _RAR_MAGICS)


def _list_rar(path: Path, issues: _Collector) -> list[str]:
    tool = shutil.which("bsdtar")
    if tool is None:
        issues.add(path.name, path.name, "TOOL_MISSING", "bsdtar is required for RAR")
        return []
    result = subprocess.run([tool, "-tf", path], capture_output=True, text=True, check=False)
    if result.returncode:
        issues.add(path.name, path.name, "RAR_CORRUPT", result.stderr.strip())
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _extract_rar(path: Path, destination: Path, issues: _Collector) -> list[Path]:
    tool = shutil.which("bsdtar")
    if tool is None:
        issues.add(path.name, path.name, "TOOL_MISSING", "bsdtar is required for RAR")
        return []
    result = subprocess.run([tool, "-xf", path, "-C", destination], capture_output=True, text=True, check=False)
    if result.returncode:
        issues.add(path.name, path.name, "RAR_EXTRACTION_FAILED", result.stderr.strip())
        return []
    return sorted(destination.rglob("*.avi"))


def _validate_archive_members(source: str, members: list[str], issues: _Collector) -> None:
    seen: set[str] = set()
    for name in members:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            issues.add(source, name, "UNSAFE_MEMBER", "absolute or parent path in RAR")
        if name in seen:
            issues.add(source, name, "DUPLICATE_MEMBER", "duplicate RAR member")
        seen.add(name)


def _probe_video(path: Path, issues: _Collector, location: str) -> tuple[str, str, int, int, float] | None:
    tool = shutil.which("ffprobe")
    if tool is None:
        issues.add(VIDEO_ARCHIVE, location, "TOOL_MISSING", "ffprobe is required")
        return None
    # JSON 输出比解析 ffprobe 的人类可读文本稳定，也便于显式选择 video stream。
    result = subprocess.run(
        [tool, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        issues.add(VIDEO_ARCHIVE, location, "VIDEO_PROBE_FAILED", result.stderr.strip())
        return None
    try:
        payload = json.loads(result.stdout)
        streams = [item for item in payload["streams"] if item.get("codec_type") == "video"]
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(stream.get("duration") or payload["format"].get("duration"))
        frame_rate = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        container = payload["format"].get("format_name", "unknown")
        codec = stream.get("codec_name", "unknown")
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        issues.add(VIDEO_ARCHIVE, location, "VIDEO_METADATA_INVALID", str(exc))
        return None
    if width < 1 or height < 1 or duration <= 0 or frame_rate <= 0:
        issues.add(VIDEO_ARCHIVE, location, "VIDEO_METADATA_INVALID", f"{width}x{height}, duration={duration}, fps={frame_rate}")
        return None
    if "avi" not in container.split(","):
        issues.add(VIDEO_ARCHIVE, location, "CONTAINER_MISMATCH", container)
    return container, codec, width, height, duration


def _decode_video(path: Path, issues: _Collector, location: str) -> bool:
    tool = shutil.which("ffmpeg")
    if tool is None:
        issues.add(VIDEO_ARCHIVE, location, "TOOL_MISSING", "ffmpeg is required")
        return False
    # null muxer 不写视频，只迫使 ffmpeg 解码完整视频流，因此不会产生中间数据。
    result = subprocess.run(
        [tool, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or result.stderr.strip():
        issues.add(VIDEO_ARCHIVE, location, "VIDEO_DECODE_FAILED", result.stderr.strip())
        return False
    return True


def _parse_classes(text: str | None, issues: _Collector) -> dict[str, int]:
    if text is None:
        issues.add(SPLIT_ARCHIVE, "classInd.txt", "MISSING_SPLIT_FILE", "class map is missing")
        return {}
    classes: dict[str, int] = {}
    ids: set[int] = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_id_text, name = line.split(maxsplit=1)
            class_id = int(class_id_text)
        except ValueError:
            issues.add(SPLIT_ARCHIVE, f"classInd.txt:{number}", "CLASS_LINE_INVALID", line)
            continue
        if name in classes or class_id in ids:
            issues.add(SPLIT_ARCHIVE, f"classInd.txt:{number}", "DUPLICATE_CLASS", line)
        classes[name] = class_id
        ids.add(class_id)
    if ids and ids != set(range(1, len(ids) + 1)):
        issues.add(SPLIT_ARCHIVE, "classInd.txt", "CLASS_IDS_NONCONTIGUOUS", "class ids must start at 1")
    return classes


def _parse_train(text: str | None, fold: int, issues: _Collector) -> dict[str, int]:
    filename = f"trainlist0{fold}.txt"
    if text is None:
        issues.add(SPLIT_ARCHIVE, filename, "MISSING_SPLIT_FILE", "train split is missing")
        return {}
    result: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            path, label = line.rsplit(maxsplit=1)
            value = int(label)
        except ValueError:
            issues.add(SPLIT_ARCHIVE, f"{filename}:{number}", "SPLIT_LINE_INVALID", line)
            continue
        if path in result:
            issues.add(SPLIT_ARCHIVE, f"{filename}:{number}", "DUPLICATE_SPLIT_ENTRY", path)
        result[path] = value
    return result


def _parse_test(text: str | None, fold: int, issues: _Collector) -> set[str]:
    filename = f"testlist0{fold}.txt"
    if text is None:
        issues.add(SPLIT_ARCHIVE, filename, "MISSING_SPLIT_FILE", "test split is missing")
        return set()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    duplicates = len(lines) - len(set(lines))
    if duplicates:
        issues.add(SPLIT_ARCHIVE, filename, "DUPLICATE_SPLIT_ENTRY", f"{duplicates} duplicates")
    return set(lines)


def _validate_fold(
    fold: int,
    train: dict[str, int],
    test: set[str],
    videos: set[str],
    classes: dict[str, int],
    issues: _Collector,
) -> None:
    # 路径级重叠是最直接的泄漏；coverage 则防止视频未被任何一侧使用。
    overlap = set(train) & test
    for path in sorted(overlap):
        issues.add(SPLIT_ARCHIVE, f"fold {fold}", "TRAIN_TEST_OVERLAP", path)
    references = set(train) | test
    for path in sorted(references - videos):
        issues.add(SPLIT_ARCHIVE, f"fold {fold}", "VIDEO_REFERENCE_MISSING", path)
    if references != videos:
        issues.add(SPLIT_ARCHIVE, f"fold {fold}", "SPLIT_COVERAGE_MISMATCH", f"split={len(references)}, archive={len(videos)}")
    # UCF101 同一 gXX 具有相近背景和视角，必须作为整体留在 train 或 test 一侧。
    groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for path in references:
        parts = PurePosixPath(path).parts
        class_name = parts[-2] if len(parts) >= 2 else ""
        match = _VIDEO_PATTERN.match(parts[-1])
        # UCF101 itself contains a known spelling/case inconsistency such as
        # HandstandPushups/ vs v_HandStandPushups_*. The directory and
        # classInd.txt are authoritative; the filename is used for gXX only.
        if class_name not in classes or match is None:
            issues.add(SPLIT_ARCHIVE, f"fold {fold}", "VIDEO_NAME_INVALID", path)
            continue
        if path in train and train[path] != classes[class_name]:
            issues.add(SPLIT_ARCHIVE, f"fold {fold}", "TRAIN_LABEL_MISMATCH", path)
        groups[(class_name, match.group("group"))].add("train" if path in train else "test")
    for key, sides in groups.items():
        if len(sides) > 1:
            issues.add(SPLIT_ARCHIVE, f"fold {fold}", "GROUP_LEAKAGE", "/".join(key))


def _validate_content_leakage(
    folds: dict[int, tuple[dict[str, int], set[str]]],
    hashes: dict[str, str],
    issues: _Collector,
) -> None:
    # 对每一折比较摘要交集，可识别改了文件名但视频字节完全相同的副本。
    for fold, (train, test) in folds.items():
        train_hashes = {hashes[path]: path for path in train if path in hashes}
        test_hashes = {hashes[path]: path for path in test if path in hashes}
        for digest in train_hashes.keys() & test_hashes.keys():
            issues.add(SPLIT_ARCHIVE, f"fold {fold}", "CONTENT_LEAKAGE", f"{train_hashes[digest]} == {test_hashes[digest]}")


def _canonical_video(value: str | Path) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    return "/".join(parts[-2:])


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            value.update(chunk)
    return value.hexdigest()


def _data_summary(
    class_count: int,
    video_count: int,
    fold_count: int,
    split_entries: int,
    digests_checked: bool,
    issues: _Collector,
) -> DataValidationSummary:
    return DataValidationSummary(
        class_count=class_count,
        video_count=video_count,
        fold_count=fold_count,
        split_entries=split_entries,
        content_digests_checked=digests_checked,
        issue_counts=dict(sorted(issues.counts.items())),
        issue_samples=tuple(issues.samples),
    )
