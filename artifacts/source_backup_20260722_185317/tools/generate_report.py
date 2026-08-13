"""Generate an offline telemetry HTML report for a run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.visualization.report_generator import generate_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate dashboard.html for a telemetry run.")
    parser.add_argument("run_dir", type=str, help="Telemetry run directory.")
    args = parser.parse_args(argv)
    result = generate_report(Path(args.run_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if bool(result.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
