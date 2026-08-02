"""Known package contracts for datasets currently fetched by TrainFoundry."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.validate.common import IssueCollector


def validate_known_package(
    root: Path,
    paths: list[Path],
    issues: IssueCollector,
) -> None:
    """Apply a deterministic contract when a known file set is discovered."""
    if not root.is_dir():
        return
    by_name = {path.name: path for path in paths if path.is_file()}
    if {"annotations_trainval2017.zip", "val2017.zip"}.issubset(by_name):
        _validate_coco2017(
            by_name["annotations_trainval2017.zip"],
            by_name["val2017.zip"],
            issues,
        )
    if {"UCF101.rar", "UCF101TrainTestSplits.zip"}.issubset(by_name):
        _validate_ucf101(
            by_name["UCF101.rar"],
            by_name["UCF101TrainTestSplits.zip"],
            issues,
        )


def _validate_coco2017(
    annotations_path: Path,
    images_path: Path,
    issues: IssueCollector,
) -> None:
    location = annotations_path.name
    try:
        with zipfile.ZipFile(images_path) as images_archive:
            image_members = {
                Path(info.filename).name
                for info in images_archive.infolist()
                if not info.is_dir()
            }
        with zipfile.ZipFile(annotations_path) as annotations_archive:
            candidates = [
                info
                for info in annotations_archive.infolist()
                if info.filename.endswith("instances_val2017.json")
            ]
            if len(candidates) != 1:
                issues.add(
                    "PACKAGE_INCOMPLETE",
                    "COCO package must contain exactly one instances_val2017.json",
                    location=location,
                )
                return
            with annotations_archive.open(candidates[0]) as stream:
                payload = json.load(stream)
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        issues.add("CONTAINER_CORRUPT", str(exc), location=location)
        return

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not all(isinstance(value, list) for value in (images, annotations, categories)):
        issues.add(
            "PACKAGE_INCOMPLETE",
            "COCO annotations require images, annotations, and categories arrays",
            location=location,
        )
        return

    image_by_id: dict[Any, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
            issues.add(
                "ANNOTATION_INVALID",
                "COCO image entry lacks id or file_name",
                location=location,
            )
            continue
        image_id = image["id"]
        if image_id in image_by_id:
            issues.add(
                "DUPLICATE_ENTRY",
                f"duplicate COCO image id: {image_id}",
                location=location,
            )
        image_by_id[image_id] = image
        if image["file_name"] not in image_members:
            issues.add(
                "REFERENCE_MISSING",
                f"COCO image is missing from val2017.zip: {image['file_name']}",
                location=images_path.name,
            )
        if image.get("width", 0) <= 0 or image.get("height", 0) <= 0:
            issues.add(
                "IMAGE_DIMENSIONS",
                f"COCO image has invalid dimensions: {image['file_name']}",
                location=location,
            )

    category_ids = {
        category.get("id") for category in categories if isinstance(category, dict)
    }
    annotation_ids: set[Any] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            issues.add(
                "ANNOTATION_INVALID",
                "COCO annotation must be an object",
                location=location,
            )
            continue
        annotation_id = annotation.get("id")
        if annotation_id in annotation_ids:
            issues.add(
                "DUPLICATE_ENTRY",
                f"duplicate COCO annotation id: {annotation_id}",
                location=location,
            )
        annotation_ids.add(annotation_id)
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            issues.add(
                "REFERENCE_MISSING",
                f"annotation {annotation_id} references an unknown image_id",
                location=location,
            )
            continue
        if annotation.get("category_id") not in category_ids:
            issues.add(
                "REFERENCE_MISSING",
                f"annotation {annotation_id} references an unknown category_id",
                location=location,
            )
        _validate_coco_geometry(annotation, image, location, issues)


def _validate_coco_geometry(
    annotation: dict[str, Any],
    image: dict[str, Any],
    location: str,
    issues: IssueCollector,
) -> None:
    annotation_id = annotation.get("id")
    bbox = annotation.get("bbox")
    if bbox is not None:
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            issues.add(
                "ANNOTATION_INVALID",
                f"annotation {annotation_id} has an invalid bbox",
                location=location,
            )
        else:
            x, y, width, height = bbox
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > image.get("width", 0) + 1
                or y + height > image.get("height", 0) + 1
            ):
                issues.add(
                    "ANNOTATION_OUT_OF_BOUNDS",
                    f"annotation {annotation_id} bbox exceeds image bounds",
                    location=location,
                )
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if (
                not isinstance(polygon, list)
                or len(polygon) < 6
                or len(polygon) % 2
                or not all(isinstance(value, (int, float)) for value in polygon)
            ):
                issues.add(
                    "ANNOTATION_INVALID",
                    f"annotation {annotation_id} has an invalid polygon",
                    location=location,
                )
                break
    keypoints = annotation.get("keypoints")
    if keypoints is not None and (
        not isinstance(keypoints, list)
        or len(keypoints) % 3
        or not all(isinstance(value, (int, float)) for value in keypoints)
    ):
        issues.add(
            "ANNOTATION_INVALID",
            f"annotation {annotation_id} has invalid keypoints",
            location=location,
        )


def _validate_ucf101(
    videos_path: Path,
    splits_path: Path,
    issues: IssueCollector,
) -> None:
    location = splits_path.name
    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        issues.add(
            "VALIDATOR_UNAVAILABLE",
            "UCF101 reference validation requires bsdtar",
            location=videos_path.name,
        )
        return
    try:
        listing = subprocess.run(
            [bsdtar, "-tf", str(videos_path)],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        videos = {
            _normalize_ucf_path(name)
            for name in listing.stdout.splitlines()
            if name.lower().endswith(".avi")
        }
        with zipfile.ZipFile(splits_path) as archive:
            text_members = {
                Path(info.filename).name: archive.read(info).decode("utf-8-sig")
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".txt")
            }
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        issues.add("CONTAINER_CORRUPT", str(exc), location=location)
        return

    class_map = _parse_ucf_classes(text_members.get("classInd.txt", ""), issues, location)
    if not class_map:
        issues.add(
            "PACKAGE_INCOMPLETE",
            "UCF101 split package lacks a valid classInd.txt",
            location=location,
        )
    folds: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"train": set(), "test": set()}
    )
    for name, text in text_members.items():
        kind, fold = _ucf_split_name(name)
        if kind is None:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            video = _normalize_ucf_path(parts[0])
            if video not in videos:
                issues.add(
                    "REFERENCE_MISSING",
                    f"split references missing video: {video}",
                    location=f"{location}!/{name}:{line_number}",
                )
            if video in folds[fold][kind]:
                issues.add(
                    "DUPLICATE_ENTRY",
                    f"duplicate split entry: {video}",
                    location=f"{location}!/{name}:{line_number}",
                )
            folds[fold][kind].add(video)
            if kind == "train" and len(parts) >= 2:
                expected_label = class_map.get(video.split("/", 1)[0])
                try:
                    actual_label = int(parts[1])
                except ValueError:
                    actual_label = None
                if expected_label is None or actual_label != expected_label:
                    issues.add(
                        "LABEL_MISMATCH",
                        f"class label does not match classInd.txt: {line}",
                        location=f"{location}!/{name}:{line_number}",
                    )
    for fold, entries in folds.items():
        for video in sorted(entries["train"] & entries["test"]):
            issues.add(
                "SPLIT_LEAKAGE",
                f"video appears in train and test for fold {fold}: {video}",
                location=location,
            )


def _parse_ucf_classes(
    text: str,
    issues: IssueCollector,
    location: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            label = int(parts[0])
        except ValueError:
            issues.add(
                "LABEL_MISMATCH",
                f"invalid classInd entry: {line}",
                location=f"{location}!/classInd.txt:{line_number}",
            )
            continue
        result[parts[1]] = label
    return result


def _ucf_split_name(name: str) -> tuple[str | None, int]:
    lowered = name.lower()
    for kind, prefix in (("train", "trainlist"), ("test", "testlist")):
        if lowered.startswith(prefix) and lowered.endswith(".txt"):
            digits = "".join(character for character in lowered if character.isdigit())
            return kind, int(digits) if digits else 0
    return None, 0


def _normalize_ucf_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("./")
    if normalized.startswith("UCF-101/"):
        normalized = normalized.removeprefix("UCF-101/")
    return normalized


__all__ = ["validate_known_package"]
