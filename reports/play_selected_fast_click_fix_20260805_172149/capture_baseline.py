from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


PROJECT = Path(r"C:\robotics_sim\wlr_robot\height_based_obstacle_replay")
REPORT = PROJECT / "reports" / "play_selected_fast_click_fix_20260805_172149"
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
    if not root.exists():
        continue
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"Path": str(path.resolve()), "Length": path.stat().st_size, "SHA256": digest(path)})

(REPORT / "protected_data_sha256_before.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
status = subprocess.run(
    ["git", "status", "--short"], cwd=PROJECT, text=True, encoding="utf-8", errors="replace",
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
).stdout
(REPORT / "git_status_before.txt").write_text(status, encoding="utf-8")
(REPORT / "baseline_note.txt").write_text(
    "Git repository: yes\n"
    "Branch: main\n"
    "Attachment source comparison: unavailable; attachment directory contains only pasted-text.txt.\n"
    "Formal project source is authoritative.\n"
    f"Protected files hashed: {len(rows)}\n",
    encoding="utf-8",
)
print(f"protected_files={len(rows)}")
print(status, end="")
