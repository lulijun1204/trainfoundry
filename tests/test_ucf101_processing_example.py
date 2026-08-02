import zipfile

from pipeline.examples.data_processing.ucf101 import processing


def _write_archives(root, *, group_leakage=False):
    root.mkdir()
    (root / "UCF101.rar").write_bytes(b"Rar!\x1a\x07\x00fake")
    second = "Class/v_Class_g01_c02.avi" if group_leakage else "Class/v_Class_g02_c01.avi"
    with zipfile.ZipFile(root / "UCF101TrainTestSplits.zip", "w") as archive:
        archive.writestr("ucfTrainTestlist/classInd.txt", "1 Class\n")
        for fold in (1, 2, 3):
            archive.writestr(
                f"ucfTrainTestlist/trainlist0{fold}.txt",
                "Class/v_Class_g01_c01.avi 1\n",
            )
            archive.writestr(f"ucfTrainTestlist/testlist0{fold}.txt", f"{second}\n")
    return second


def _members(second):
    return [
        "UCF-101/Class/v_Class_g01_c01.avi",
        f"UCF-101/{second}",
    ]


def test_ucf_file_validation_probes_and_fully_decodes_every_video(tmp_path, monkeypatch):
    root = tmp_path / "ucf101"
    second = _write_archives(root)
    monkeypatch.setattr(processing, "_list_rar", lambda path, issues: _members(second))

    def extract(path, destination, issues):
        files = []
        for member in _members(second):
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video")
            files.append(target)
        return files

    monkeypatch.setattr(processing, "_extract_rar", extract)
    monkeypatch.setattr(
        processing,
        "_probe_video",
        lambda path, issues, location: ("avi", "mpeg4", 320, 240, 4.0),
    )
    monkeypatch.setattr(processing, "_decode_video", lambda path, issues, location: True)

    summary = processing.validate_file_format(root, expected_video_count=None)

    assert summary.valid_archives == 2
    assert summary.video_files == 2
    assert summary.probed_videos == 2
    assert summary.fully_decoded_videos == 2
    assert summary.video_codecs == {"mpeg4": 2}
    assert summary.issue_counts == {}


def test_ucf_data_validation_accepts_official_group_isolation(tmp_path, monkeypatch):
    root = tmp_path / "ucf101"
    second = _write_archives(root)
    monkeypatch.setattr(processing, "_list_rar", lambda path, issues: _members(second))

    summary = processing.validate_dataset(
        root,
        strict_official_counts=False,
        check_content_duplicates=False,
    )

    assert summary.class_count == 1
    assert summary.video_count == 2
    assert summary.fold_count == 3
    assert summary.split_entries == 6
    assert summary.issue_counts == {}


def test_ucf_data_validation_detects_group_leakage(tmp_path, monkeypatch):
    root = tmp_path / "ucf101"
    second = _write_archives(root, group_leakage=True)
    monkeypatch.setattr(processing, "_list_rar", lambda path, issues: _members(second))

    summary = processing.validate_dataset(
        root,
        strict_official_counts=False,
        check_content_duplicates=False,
    )

    assert summary.issue_counts == {"GROUP_LEAKAGE": 3}
