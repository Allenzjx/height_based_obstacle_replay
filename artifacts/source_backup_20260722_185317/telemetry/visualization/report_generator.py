"""Offline report generation for telemetry run directories."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from telemetry.exporters import read_csv_rows, write_json


def generate_report(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    samples = read_csv_rows(root / "telemetry_samples.csv")
    joints = read_csv_rows(root / "joint_timeseries.csv")
    contacts = read_csv_rows(root / "contacts.csv")
    events = _read_jsonl(root / "events.jsonl")
    summary = _read_json(root / "stability_summary.json")
    metadata = _read_json(root / "metadata.json")
    figure_paths = _generate_figures(figures, samples, joints, contacts)
    html_path = root / "dashboard.html"
    html_path.write_text(
        _html_document(
            run_dir=root,
            samples=samples,
            joints=joints,
            contacts=contacts,
            events=events,
            summary=summary,
            metadata=metadata,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    result = {
        "ok": True,
        "run_dir": str(root),
        "dashboard": str(html_path),
        "figures": [str(path) for path in figure_paths],
        "plot_backend": "matplotlib" if figure_paths else "html_only",
    }
    write_json(root / "report_status.json", result)
    return result


def _generate_figures(
    figures: Path,
    samples: list[dict[str, str]],
    joints: list[dict[str, str]],
    contacts: list[dict[str, str]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []
    paths: list[Path] = []
    time_s = _series(samples, "time_s")
    if time_s:
        path = figures / "summary.png"
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(time_s, _series(samples, "static_stability_margin_m"), label="static")
        axes[0].plot(time_s, _series(samples, "dynamic_stability_margin_m"), label="dynamic")
        axes[0].plot(time_s, _series(samples, "equilibrium_stability_margin_m"), label="equilibrium")
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_ylabel("margin m")
        axes[0].legend(loc="best")
        axes[1].plot(time_s, _series(samples, "base_roll_rad"), label="roll")
        axes[1].plot(time_s, _series(samples, "base_pitch_rad"), label="pitch")
        axes[1].set_ylabel("base rad")
        axes[1].legend(loc="best")
        axes[2].plot(time_s, _series(samples, "com_x_m"), label="com x")
        axes[2].plot(time_s, _series(samples, "com_y_m"), label="com y")
        axes[2].set_ylabel("COM m")
        axes[2].set_xlabel("time s")
        axes[2].legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

        path = figures / "stability_report.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(_series(samples, "com_projection_x_m"), _series(samples, "com_projection_y_m"), label="COM projection")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("support x m")
        ax.set_ylabel("support y m")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    if joints:
        path = figures / "joint_report.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in _top_joint_names(joints, limit=8):
            rows = [row for row in joints if row.get("joint_name") == name]
            ax.plot(_series(rows, "time_s"), _series(rows, "torque_utilization"), label=name)
        ax.set_xlabel("time s")
        ax.set_ylabel("torque utilization")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    if contacts:
        path = figures / "contact_report.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in _top_body_names(contacts, limit=8):
            rows = [row for row in contacts if row.get("body_name") == name]
            ax.plot(_series(rows, "time_s"), _series(rows, "normal_force_n"), label=name)
        ax.set_xlabel("time s")
        ax.set_ylabel("normal force N")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def _html_document(
    *,
    run_dir: Path,
    samples: list[dict[str, str]],
    joints: list[dict[str, str]],
    contacts: list[dict[str, str]],
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    title = html.escape(str(metadata.get("run_label", run_dir.name)))
    cards = [
        ("Samples", len(samples)),
        ("Joint Rows", len(joints)),
        ("Contact Rows", len(contacts)),
        ("Events", len(events)),
        ("Min Static Margin", _fmt(summary.get("min_static_margin_m"), "m")),
        ("Min Dynamic Margin", _fmt(summary.get("min_dynamic_margin_m"), "m")),
        ("Max Torque Util", _fmt(summary.get("max_torque_utilization"), "")),
        ("Mean Overhead", _fmt(summary.get("mean_sample_overhead_ms"), "ms")),
    ]
    figures_html = "\n".join(f'<section><img src="{html.escape(_rel(run_dir, path))}" alt="{html.escape(path.stem)}"></section>' for path in figure_paths)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; background: #f7f7f4; color: #202124; }}
h1 {{ font-size: 24px; margin: 0 0 16px; }}
h2 {{ font-size: 18px; margin-top: 28px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
.metric {{ border: 1px solid #d7d7d0; border-radius: 8px; padding: 12px; background: white; }}
.metric b {{ display: block; font-size: 12px; color: #5d6368; font-weight: 600; }}
.metric span {{ font-size: 20px; }}
img {{ max-width: 100%; border: 1px solid #d7d7d0; background: white; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; text-align: left; vertical-align: top; }}
th {{ background: #efefea; }}
code {{ background: #eeeeea; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">{''.join(_metric_card(name, value) for name, value in cards)}</div>
<h2>Figures</h2>
{figures_html or '<p>No figures were generated; check matplotlib availability or sample files.</p>'}
<h2>Stability Summary</h2>
{_summary_table(summary)}
<h2>Recent Events</h2>
{_events_table(events[-80:])}
<h2>Files</h2>
<table><tbody>
<tr><th>Run Directory</th><td><code>{html.escape(str(run_dir))}</code></td></tr>
<tr><th>Telemetry CSV</th><td><code>telemetry_samples.csv</code></td></tr>
<tr><th>Body COM CSV</th><td><code>body_com_timeseries.csv</code></td></tr>
<tr><th>Joint CSV</th><td><code>joint_timeseries.csv</code></td></tr>
<tr><th>Contacts CSV</th><td><code>contacts.csv</code></td></tr>
<tr><th>Events JSONL</th><td><code>events.jsonl</code></td></tr>
<tr><th>Model Audit</th><td><code>model_audit.json</code></td></tr>
</tbody></table>
</body>
</html>
"""


def _metric_card(name: str, value: Any) -> str:
    return f'<div class="metric"><b>{html.escape(str(name))}</b><span>{html.escape(str(value))}</span></div>'


def _summary_table(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, ensure_ascii=False, default=str))}</td></tr>"
        for key, value in summary.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _events_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return "<p>No events recorded.</p>"
    rows = []
    for event in events:
        rows.append(
            "<tr>"
            f"<td>{html.escape(_fmt(event.get('simulation_time_s'), 's'))}</td>"
            f"<td>{html.escape(str(event.get('severity', '')))}</td>"
            f"<td>{html.escape(str(event.get('event_type', '')))}</td>"
            f"<td>{html.escape(str(event.get('message', '')))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Time</th><th>Severity</th><th>Type</th><th>Message</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_float(row.get(key)) for row in rows]


def _top_joint_names(rows: list[dict[str, str]], *, limit: int) -> list[str]:
    scores: dict[str, float] = {}
    for row in rows:
        name = str(row.get("joint_name", ""))
        scores[name] = max(scores.get(name, 0.0), abs(_float(row.get("torque_utilization"))))
    return [name for name, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit] if name]


def _top_body_names(rows: list[dict[str, str]], *, limit: int) -> list[str]:
    scores: dict[str, float] = {}
    for row in rows:
        name = str(row.get("body_name", ""))
        scores[name] = max(scores.get(name, 0.0), abs(_float(row.get("normal_force_n"))))
    return [name for name, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit] if name]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        except Exception:
            pass
    return rows


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any, suffix: str) -> str:
    number = _float(value)
    if math.isfinite(number):
        return f"{number:.4g}{suffix}"
    return str(value)


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")
