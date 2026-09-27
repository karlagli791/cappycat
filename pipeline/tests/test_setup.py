"""What the installer's AI-setup wizard relies on: ``doctor --json`` (``ready`` / ``missing`` /
``packages``) and ``download-models`` progress with byte counts (idempotent)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cappycat_pipeline import downloads, models

PIPELINE_DIR = Path(__file__).resolve().parent.parent


def _run(args, env_extra=None, timeout=300):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "cappycat_pipeline", *args], cwd=str(PIPELINE_DIR), capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout, env=env)


def test_doctor_json_reports_missing_models(tmp_path):
    empty = tmp_path / "models"
    env = {"CAPPYCAT_MODELS_DIR": str(empty), "HF_HOME": str(empty / "hf"), "TORCH_HOME": str(empty / "torch")}
    proc = _run(["doctor", "--json"], env)
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(proc.stdout)  # exactly one JSON object
    # the existing keys are still there
    for k in ("python", "ffmpeg", "torch", "torchCuda", "device", "vram", "models", "modelsReady", "shotBackend",
              "flowBackend", "mode", "demucs", "separation"):
        assert k in rep, k
    assert isinstance(rep["ready"], bool) and isinstance(rep["missing"], list) and isinstance(rep["warnings"], list)
    assert rep["ready"] is False and rep["modelsReady"] is False
    assert {f"model:{m}" for m in rep["models"]} <= set(rep["missing"])
    assert "model:raft_small" in rep["missing"]
    assert set(rep["packages"]) >= {"torch", "torchvision", "numpy", "opencv", "ultralytics", "demucs"}
    assert rep["packages"]["numpy"]
    assert rep["ready"] == (not rep["missing"])
    assert all(m == "ffmpeg" or m.startswith(("package:", "model:")) for m in rep["missing"])


def test_doctor_ready_with_the_dev_install():
    if not all(m.present for m in models.model_status()):
        pytest.skip("models not downloaded")
    rep = json.loads(_run(["doctor"]).stdout)
    assert rep["ready"] == (not rep["missing"])
    assert not [m for m in rep["missing"] if m.startswith("model:")]


def test_download_all_reports_bytes_and_is_idempotent(monkeypatch, tmp_path):
    target = tmp_path / "fake"
    total = 3 * 400_000

    def fake_download():
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "w.bin.incomplete", "wb") as f:
            for _ in range(3):
                f.write(b"\0" * 400_000)
                f.flush()
                time.sleep(0.35)
        os.replace(target / "w.bin.incomplete", target / "w.bin")

    monkeypatch.setattr(downloads, "steps", lambda: [("fake-model", fake_download)])
    monkeypatch.setattr(downloads, "_present", lambda name: (target / "w.bin").is_file())
    monkeypatch.setattr(downloads, "watch_paths", lambda name: [(target, "*")])
    monkeypatch.setitem(downloads.EXPECTED_BYTES, "fake-model", total)
    monkeypatch.setattr(downloads, "WATCH_INTERVAL_S", 0.1)
    events = []

    def prog(name, pct, msg, extra):
        events.append((name, pct, msg, dict(extra)))

    res = downloads.download_all(prog)
    assert res == [("fake-model", None)]
    assert all(e[0] == "fake-model" for e in events)
    pcts = [e[1] for e in events]
    assert pcts == sorted(pcts) and pcts[-1] == 1.0
    got = [e[3]["bytes"] for e in events]
    assert got == sorted(got) and got[-1] == total
    assert any(0 < b < total for b in got)  # intermediate byte counts while downloading
    assert all(e[3]["totalBytes"] == total and e[3]["overallTotalBytes"] == total for e in events)
    assert "ok" in events[-1][2]
    # second run: nothing to download
    events.clear()
    assert downloads.download_all(prog) == [("fake-model", None)]
    assert len(events) == 1 and "already present" in events[0][2] and events[0][1] == 1.0
    # old-style 3-argument callbacks still work (the analyze first-run path)
    calls = []
    downloads.download_all(lambda n, p, m: calls.append((n, p, m)))
    assert calls and len(calls[0]) == 3


def test_download_models_cli_events_carry_bytes():
    if models.raft_small_path() is None:
        pytest.skip("RAFT weights not downloaded")
    proc = _run(["download-models", "--only", "raft"])
    assert proc.returncode == 0, proc.stderr
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    prog = [e for e in events if e["event"] == "progress"]
    assert prog and all(e["stage"] == "download" and {"bytes", "totalBytes", "overallBytes", "overallTotalBytes"} <= set(e)
                        for e in prog)
    assert prog[-1]["pct"] == 1.0 and "already present" in prog[-1]["message"]
    assert events[-1]["event"] == "result"
