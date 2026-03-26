from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

from us_invest_ai.repo_health import write_repo_health_snapshot


def _fresh_dir(name: str) -> Path:
    root = Path("test_runtime_temp") / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_write_repo_health_snapshot_records_success(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b"## main\n M foo.py\n", stderr=b"")

    monkeypatch.setattr("us_invest_ai.repo_health.subprocess.run", fake_run)
    root = _fresh_dir("repo_health_success")
    output_path = root / "repo_health.txt"

    payload = write_repo_health_snapshot(root, output_path)

    assert payload["status"] == "succeeded"
    assert payload["exit_code"] == 0
    assert payload["last_output"] == " M foo.py"
    assert output_path.read_text(encoding="utf-8").startswith("## main")


def test_write_repo_health_snapshot_records_warning_without_raising(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=128, stdout=b"", stderr=b"fatal: unsafe repo")

    monkeypatch.setattr("us_invest_ai.repo_health.subprocess.run", fake_run)
    root = _fresh_dir("repo_health_warning")
    output_path = root / "repo_health.txt"

    payload = write_repo_health_snapshot(root, output_path)

    assert payload["status"] == "warning"
    assert payload["exit_code"] == 128
    assert payload["last_output"] == "fatal: unsafe repo"
    assert "[repo-health warning]" in output_path.read_text(encoding="utf-8")
