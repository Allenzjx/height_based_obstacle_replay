from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "selected_pose_run_management_and_full_replay_fix_20260805_235722"
PROTECTED_ROOTS = (
    ROOT / "saved_height_steps",
    ROOT / "saved_height_steps_fsm_reference_v1",
    ROOT / "saved_height_steps_fsm_reference_v2",
    ROOT / "saved_sequences",
)
PROTECTED_FILES = (ROOT / "history_commands.log",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_tree() -> dict[str, dict[str, object]]:
    files: list[Path] = []
    for root in PROTECTED_ROOTS:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(path for path in PROTECTED_FILES if path.is_file())
    reports_root = ROOT / "reports"
    if reports_root.exists():
        files.extend(
            path
            for path in reports_root.rglob("*")
            if path.is_file() and REPORT not in path.parents
        )
    return {
        str(path.relative_to(ROOT)): {
            "length": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(set(files))
    }


def git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


REPORT.mkdir(parents=True, exist_ok=True)
(REPORT / "protected_data_sha256_before.json").write_text(
    json.dumps(capture_tree(), indent=2, sort_keys=True), encoding="utf-8"
)
(REPORT / "git_baseline.txt").write_text(
    "\n".join(
        (
            f"HEAD={git('rev-parse', 'HEAD')}",
            f"origin/main={git('rev-parse', 'origin/main')}",
            f"ahead_behind={git('rev-list', '--left-right', '--count', 'HEAD...origin/main')}",
            "status_short_at_start:",
            git("status", "--short"),
            "remotes:",
            git("remote", "-v"),
        )
    )
    + "\n",
    encoding="utf-8",
)
print(f"report={REPORT}")
print(f"protected_file_count={len(capture_tree())}")
