from pathlib import Path

from football_tracking.data.yolo_exporter import export_yolo_dataset


def _write_sequence(root: Path) -> None:
    sequence = root / "SNMOT-TEST"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir(parents=True)
    (sequence / "img1" / "000001.jpg").write_bytes(b"test")
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-TEST\nimDir=img1\nframeRate=25\n"
        "seqLength=1\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )
    (sequence / "gameinfo.ini").write_text(
        "[Sequence]\nname=SNMOT-TEST\ngameID=1\n"
        "trackletID_1= player team left;10\n",
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,914,855,55,172,1,-1,-1,-1\n", encoding="utf-8"
    )


def test_export_yolo_dataset(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "yolo"
    _write_sequence(source)
    summary = export_yolo_dataset(source, output, image_mode="copy")
    label = output / "labels" / "train" / "SNMOT-TEST_000001.txt"
    assert summary["num_sequences"] == 1
    assert label.exists()
    assert label.read_text(encoding="utf-8") == "1 0.49036458 0.87129630 0.02864583 0.15925926\n"
    assert (output / "dataset.yaml").exists()


def test_export_skips_zero_area_annotation(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "yolo"
    _write_sequence(source)
    gt = source / "SNMOT-TEST" / "gt" / "gt.txt"
    gt.write_text(
        "1,1,914,855,0,172,1,-1,-1,-1\n", encoding="utf-8"
    )
    summary = export_yolo_dataset(source, output, image_mode="copy")
    label = output / "labels" / "train" / "SNMOT-TEST_000001.txt"
    assert summary["skipped_invalid_annotations"] == 1
    assert label.read_text(encoding="utf-8") == ""

