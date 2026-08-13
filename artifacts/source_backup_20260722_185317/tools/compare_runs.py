"""Compare telemetry summaries across run directories."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


SUMMARY_KEYS = [
    "sample_count",
    "duration_sim_s",
    "min_static_margin_m",
    "min_dynamic_margin_m",
    "min_equilibrium_margin_m",
    "max_contact_force_n",
    "max_torque_utilization",
    "event_count",
    "mean_sample_overhead_ms",
]


def compare_runs(run_dirs: list[str | Path], *, output_dir: str | Path | None = None) -> dict[str, Any]:
    rows = [_row(Path(run_dir)) for run_dir in run_dirs]
    destination = Path(output_dir) if output_dir is not None else Path(run_dirs[0]).resolve().parent / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "comparison.csv"
    html_path = destination / "comparison.html"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_dir", "label", *SUMMARY_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    html_path.write_text(_html(rows), encoding="utf-8")
    result = {"ok": True, "output_dir": str(destination), "csv": str(csv_path), "html": str(html_path), "runs": rows}
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare height replay telemetry runs.")
    parser.add_argument("run_dirs", nargs="+", type=str, help="Telemetry run directories.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for comparison.csv/html.")
    args = parser.parse_args(argv)
    result = compare_runs(args.run_dirs, output_dir=args.output_dir or None)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if bool(result.get("ok", False)) else 1


def _row(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "stability_summary.json")
    metadata = _read_json(run_dir / "metadata.json")
    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "label": str(metadata.get("run_label", run_dir.name)),
    }
    for key in SUMMARY_KEYS:
        row[key] = summary.get(key, "")
    return row


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _html(rows: list[dict[str, Any]]) -> str:
    header = "".join(f"<th>{html.escape(key)}</th>" for key in ["label", *SUMMARY_KEYS])
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_fmt(row.get(key)))}</td>" for key in ["label", *SUMMARY_KEYS])
        + "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Telemetry Run Comparison</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; background: #f7f7f4; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; text-align: left; }}
th {{ background: #efefea; }}
</style>
</head>
<body>
<h1>Telemetry Run Comparison</h1>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
"""


def _fmt(value: Any) -> str:
    try:
        number = float(value)
        if math.isfinite(number):
            return f"{number:.6g}"
    except Exception:
        pass
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
