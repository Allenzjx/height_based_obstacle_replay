from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(r"C:\robotics_sim\wlr_robot\height_based_obstacle_replay")
REPORT = PROJECT / "reports" / "fix_50mm_manual_replay_and_selected_fast_pose_restore_20260805_193824"
PROTECTED_ROOTS = [
    PROJECT / "saved_height_steps",
    PROJECT / "saved_height_steps_fsm_reference_v1",
    PROJECT / "saved_height_steps_fsm_reference_v2",
    PROJECT / "saved_sequences",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


rows = []
for root in PROTECTED_ROOTS:
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rows.append(
                {
                    "path": str(path.resolve()),
                    "length": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
(REPORT / "protected_data_sha256_before.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

commands = {
    "git_status_short": ["git", "status", "--short"],
    "git_head": ["git", "rev-parse", "HEAD"],
    "git_origin_main": ["git", "rev-parse", "origin/main"],
    "git_remote": ["git", "remote", "-v"],
}
captured = {}
for name, command in commands.items():
    captured[name] = subprocess.run(
        command,
        cwd=PROJECT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout
(REPORT / "git_baseline.txt").write_text(
    "\n".join(f"[{name}]\n{value.rstrip()}" for name, value in captured.items()) + "\n",
    encoding="utf-8",
)
print(f"protected_files={len(rows)}")
print(captured["git_status_short"], end="")
