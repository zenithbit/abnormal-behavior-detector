import json

from abd.cli import main


def test_index_and_split(toy_dataset, tmp_path, capsys):
    out = tmp_path / "out"
    common = ["--set", f"paths.dataset_root={toy_dataset}", "--set", f"paths.work_dir={out}"]
    main(common + ["index"])
    assert "TOTAL" in capsys.readouterr().out
    main(common + ["split"])
    split = json.loads((out / "splits.json").read_text())
    assert split["behavior_classes"] == ["Fighting", "Theft"]
    assert sum(len(split[k]) for k in ("train", "val", "test")) == 18
