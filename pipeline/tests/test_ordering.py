import json
from pathlib import Path

import pytest

from cappycat_pipeline.ordering import order_folder, order_paths, parse_name


def names(files, shuffle=True):
    paths = [Path(f) for f in files]
    if shuffle:
        paths = list(reversed(paths))
    infos, warnings = order_paths(paths)
    return [i.name for i in infos], warnings, infos


@pytest.mark.parametrize(
    "expected",
    [
        ["clip1.mp4", "clip2.mp4", "clip10.mp4", "clip11.mp4"],
        ["01_intro.mp4", "02_kitchen.mp4", "10_chase.mp4"],
        ["1-raccoon wakes.mp4", "2-raccoon eats.mp4", "3-raccoon naps.mp4"],
        ["Part I.mp4", "Part II.mp4", "Part III.mp4", "Part IV.mp4", "Part IX.mp4"],
        ["scene1_shot1.mp4", "scene1_shot2.mp4", "scene1_shot10.mp4", "scene2_shot1.mp4"],
        ["S01E02.mp4", "S01E10.mp4", "S02E01.mp4"],
        ["first.mp4", "second.mp4", "third.mp4", "fourth.mp4"],
        ["the 1st bit.mp4", "the 2nd bit.mp4", "the 3rd bit.mp4", "the 11th bit.mp4"],
        ["raccoon_kitchen_3.mp4", "raccoon_garden_7.mp4", "raccoon_roof_12.mp4"],
        ["Clip (1).mp4", "Clip (2).mp4", "Clip (10).mp4"],
        ["shot 3a.mp4", "shot 3b.mp4", "shot 4.mp4"],
        ["chapter one.mp4", "chapter two.mp4", "chapter three.mp4"],
        ["#1 hook.mp4", "#2 build.mp4", "#3 payoff.mp4"],
        ["kling_20260924_101500.mp4", "kling_20260924_153012.mp4", "kling_20260925_090000.mp4"],
        ["scene2shot3.mp4", "scene2shot4.mp4", "scene3shot1.mp4"],
    ],
)
def test_orders(expected):
    got, _, _ = names(expected)
    assert got == expected


def test_position_words_bracket_numbered_clips():
    got, _, infos = names(["intro.mp4", "02.mp4", "05.mp4", "11.mp4", "credits.mp4"])
    assert got == ["intro.mp4", "02.mp4", "05.mp4", "11.mp4", "credits.mp4"]
    assert "first" in infos[0].reason and "last" in infos[-1].reason


def test_ignores_resolution_version_and_random_ids():
    got, _, infos = names([
        "02_opening_1080p_v3_a3f9c2e1b7.mp4",
        "03_chase_4k_v1_x7k2m9q4p1z8.mp4",
        "10_ending_24fps_1920x1080.mp4",
    ])
    assert got == ["02_opening_1080p_v3_a3f9c2e1b7.mp4", "03_chase_4k_v1_x7k2m9q4p1z8.mp4", "10_ending_24fps_1920x1080.mp4"]
    assert infos[0].reason.startswith("leading number 2")


def test_gap_and_tie_warnings():
    _, warnings, _ = names(["01.mp4", "02.mp4", "04.mp4"])
    assert any("skips 3" in w for w in warnings)
    _, warnings, _ = names(["clip 2 a.mp4", "clip2 a.mp4", "clip 3.mp4"])
    assert any("same position" in w for w in warnings)


def test_unnumbered_falls_back_to_natural_with_warning():
    got, warnings, _ = names(["banana.mp4", "apple.mp4", "cherry.mp4"])
    assert got == ["apple.mp4", "banana.mp4", "cherry.mp4"]
    assert any("no consistent numbering" in w for w in warnings)


def test_parse_details():
    p = parse_name(Path("Ep 2 - Scene IV - shot 03b.mp4"))
    assert p.markers == {1: 2.0, 4: 4.0, 6: 3.0}
    assert p.suffix == "b"


def test_order_folder_and_cli(tmp_path, capsys):
    for n in ["3 finale.mp4", "1 open.mp4", "2 middle.mp4", "look.cube", "notes.txt", ".hidden.mp4"]:
        (tmp_path / n).write_bytes(b"x")
    infos, _ = order_folder(tmp_path)
    assert [i.name for i in infos] == ["1 open.mp4", "2 middle.mp4", "3 finale.mp4", "look.cube"]

    from cappycat_pipeline.cli import main

    assert main(["order", str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [f["name"] for f in out["files"]] == ["1 open.mp4", "2 middle.mp4", "3 finale.mp4", "look.cube"]
    assert out["files"][0]["reason"].startswith("leading number 1")
