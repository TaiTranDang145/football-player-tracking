from pathlib import Path

from football_tracking.data.validator import validate_dataset


def _write_sequence(root: Path, gt_row: str) -> None:
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
    (sequence / "gt" / "gt.txt").write_text(gt_row, encoding="utf-8")


def test_validator_accepts_valid_sequence(tmp_path: Path):
    _write_sequence(tmp_path, "1,1,10,20,50,100,1,-1,-1,-1\n")
    report = validate_dataset(tmp_path)
    assert report["valid"] is True
    assert report["num_sequences"] == 1
    assert report["class_counts"] == {"player": 1}


def test_validator_reports_invalid_bbox(tmp_path: Path):
    _write_sequence(tmp_path, "1,1,10,20,0,100,1,-1,-1,-1\n")
    report = validate_dataset(tmp_path)
    assert report["valid"] is False
    assert report["error_count"] == 1
    assert "width/height" in report["issues"][0]["message"]

