"""Pure Python Isaac interpreter preflight.

This module intentionally does not create ``SimulationApp`` at import time.
The parent process executes this file in a candidate interpreter and reads one
single-line JSON report from stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback as traceback_module
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from worker_startup_diagnostics import classify_startup_error, summarize_environment


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_ISAACLAB_ROOT = Path("C:/robotics_sim/IsaacLab")
ENVIRONMENT_KEYS = (
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "HEADLESS",
    "LIVESTREAM",
    "ENABLE_CAMERAS",
    "OMNI_KIT_ACCEPT_EULA",
    "ISAAC_PATH",
    "EXP_PATH",
    "CARB_APP_PATH",
)
EULA_PROMPT_NEEDLES = (
    "do you accept the eula",
    "accept the eula",
    "omniverse eula",
)


@dataclass
class IsaacInterpreterReport:
    executable: str = ""
    python_version: str = ""
    isaacsim_importable: bool = False
    isaaclab_importable: bool = False
    app_launcher_importable: bool = False
    isaacsim_version: str = ""
    isaaclab_version: str = ""
    compatible_python: bool = False
    eula_required_or_unknown: bool = False
    environment: dict[str, str] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""
    error_category: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_launcher_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    return summarize_environment(env or os.environ)


def inspect_python_interpreter(
    python_exe: str | Path,
    *,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
    isaaclab_root: str | Path = DEFAULT_ISAACLAB_ROOT,
) -> IsaacInterpreterReport:
    return run_interpreter_preflight(
        python_exe,
        timeout_s=timeout_s,
        env=env,
        isaaclab_root=isaaclab_root,
    )


def run_interpreter_preflight(
    python_exe: str | Path,
    *,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
    isaaclab_root: str | Path = DEFAULT_ISAACLAB_ROOT,
) -> IsaacInterpreterReport:
    """Run preflight in an independent Python process and parse JSON output."""

    started = time.monotonic()
    exe = Path(python_exe)
    if not exe.exists():
        return IsaacInterpreterReport(
            executable=str(exe),
            environment=inspect_launcher_environment(env),
            error=f"Python executable does not exist: {exe}",
            error_category="missing_python",
            elapsed_s=0.0,
        )
    child_env = build_preflight_env(env=env, isaaclab_root=isaaclab_root)
    command = [
        str(exe),
        "-u",
        str(Path(__file__).resolve()),
        "--preflight-child",
        "--isaaclab-root",
        str(isaaclab_root),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(MODULE_ROOT),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        text = _join_output(exc.stdout, exc.stderr)
        prompt = detect_eula_prompt(text)
        return IsaacInterpreterReport(
            executable=str(exe),
            environment=inspect_launcher_environment(child_env),
            eula_required_or_unknown=prompt,
            error=(
                "Isaac preflight timed out after "
                f"{float(timeout_s):g}s"
                + (" while waiting for EULA input." if prompt else ".")
            ),
            traceback=text,
            error_category="eula_required" if prompt else "startup_timeout",
            elapsed_s=max(0.0, time.monotonic() - started),
        )
    except Exception as exc:
        return IsaacInterpreterReport(
            executable=str(exe),
            environment=inspect_launcher_environment(child_env),
            error=f"Could not run interpreter preflight: {exc}",
            traceback=traceback_module.format_exc(),
            error_category=classify_startup_error(str(exc)),
            elapsed_s=max(0.0, time.monotonic() - started),
        )

    combined = _join_output(completed.stdout, completed.stderr)
    prompt = detect_eula_prompt(combined)
    report = _parse_report_from_output(completed.stdout)
    if report is None:
        report = IsaacInterpreterReport(
            executable=str(exe),
            environment=inspect_launcher_environment(child_env),
            error=f"Preflight did not produce JSON (returncode {completed.returncode}).",
            traceback=combined,
            error_category="eula_required" if prompt else classify_startup_error(combined),
        )
    if prompt:
        report.eula_required_or_unknown = True
        if not report.error:
            report.error = "Isaac preflight encountered an EULA prompt."
        report.error_category = "eula_required"
    if completed.returncode != 0 and not report.error:
        report.error = f"Preflight child exited with returncode {completed.returncode}."
        report.error_category = classify_startup_error(combined)
        report.traceback = combined
    report.elapsed_s = max(report.elapsed_s, max(0.0, time.monotonic() - started))
    if not report.error_category:
        report.error_category = classify_startup_error(report.error or report.traceback)
    return report


def run_isaaclab_bat_preflight(
    isaaclab_bat: str | Path,
    *,
    timeout_s: float = 30.0,
    env: dict[str, str] | None = None,
    isaaclab_root: str | Path = DEFAULT_ISAACLAB_ROOT,
) -> IsaacInterpreterReport:
    """Run preflight through ``isaaclab.bat -p`` without using ``-c``."""

    started = time.monotonic()
    bat = Path(isaaclab_bat)
    if not bat.exists():
        return IsaacInterpreterReport(
            executable=str(bat),
            environment=inspect_launcher_environment(env),
            error=f"IsaacLab batch launcher does not exist: {bat}",
            error_category="missing_python",
        )
    child_env = build_preflight_env(env=env, isaaclab_root=isaaclab_root)
    wrapper_dir = Path(tempfile.gettempdir()) / "height_replay_worker_configs"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / f"preflight_{int(time.time() * 1000)}.cmd"
    command_text = " ".join(
        [
            "call",
            quote_windows_cmd_arg(str(bat)),
            "-p",
            quote_windows_cmd_arg(str(Path(__file__).resolve())),
            "--preflight-child",
            "--isaaclab-root",
            quote_windows_cmd_arg(str(isaaclab_root)),
        ]
    )
    wrapper.write_text("@echo off\r\nsetlocal EnableExtensions\r\n" + command_text + "\r\nexit /b %ERRORLEVEL%\r\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(wrapper)],
            cwd=str(Path(isaaclab_root)),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        text = _join_output(exc.stdout, exc.stderr)
        prompt = detect_eula_prompt(text)
        return IsaacInterpreterReport(
            executable=str(bat),
            environment=inspect_launcher_environment(child_env),
            eula_required_or_unknown=prompt,
            error=f"IsaacLab batch preflight timed out after {float(timeout_s):g}s.",
            traceback=text,
            error_category="eula_required" if prompt else "startup_timeout",
            elapsed_s=max(0.0, time.monotonic() - started),
        )
    combined = _join_output(completed.stdout, completed.stderr)
    prompt = detect_eula_prompt(combined)
    report = _parse_report_from_output(completed.stdout)
    if report is None:
        report = IsaacInterpreterReport(
            executable=str(bat),
            environment=inspect_launcher_environment(child_env),
            error=f"IsaacLab batch preflight did not produce JSON (returncode {completed.returncode}).",
            traceback=combined,
            error_category="eula_required" if prompt else classify_startup_error(combined),
        )
    if prompt:
        report.eula_required_or_unknown = True
        report.error_category = "eula_required"
        if not report.error:
            report.error = "IsaacLab batch preflight encountered an EULA prompt."
    if completed.returncode != 0 and not report.error:
        report.error = f"IsaacLab batch preflight exited with returncode {completed.returncode}."
        report.error_category = classify_startup_error(combined)
        report.traceback = combined
    report.elapsed_s = max(report.elapsed_s, max(0.0, time.monotonic() - started))
    if not report.error_category:
        report.error_category = classify_startup_error(report.error or report.traceback)
    try:
        wrapper.unlink(missing_ok=True)
    except Exception:
        pass
    return report


def validate_isaac_python_compatibility(python_version: str, isaacsim_version: str) -> tuple[bool, str]:
    py_major, py_minor = _parse_python_major_minor(python_version)
    isaac_major = _parse_major(isaacsim_version)
    if isaac_major == 5:
        ok = (py_major, py_minor) == (3, 11)
        return ok, "" if ok else f"Isaac Sim 5.x requires Python 3.11; selected Python is {py_major}.{py_minor}."
    if isaac_major == 4:
        ok = (py_major, py_minor) == (3, 10)
        return ok, "" if ok else f"Isaac Sim 4.x requires Python 3.10; selected Python is {py_major}.{py_minor}."
    if isaac_major <= 0:
        return False, "Could not determine Isaac Sim major version."
    return True, ""


def format_preflight_error(report: IsaacInterpreterReport | dict[str, Any]) -> str:
    data = report.to_dict() if isinstance(report, IsaacInterpreterReport) else dict(report)
    if data.get("eula_required_or_unknown") and data.get("error_category") == "eula_required":
        return (
            "Isaac Sim requested EULA input. Background workers cannot answer prompts; "
            "run Isaac Sim once in the foreground or pass --accept-isaac-eula only if you explicitly agree."
        )
    if not data.get("isaacsim_importable"):
        return f"Selected Python cannot import isaacsim: {data.get('error') or data.get('traceback') or data.get('executable')}"
    if not data.get("isaaclab_importable"):
        return f"Selected Python cannot import isaaclab: {data.get('error') or data.get('traceback') or data.get('executable')}"
    if not data.get("app_launcher_importable"):
        return f"Selected Python cannot import isaaclab.app.AppLauncher: {data.get('error') or data.get('traceback')}"
    if not data.get("compatible_python"):
        ok, reason = validate_isaac_python_compatibility(
            str(data.get("python_version", "")),
            str(data.get("isaacsim_version", "")),
        )
        return reason if not ok else "Isaac Python compatibility check failed."
    return str(data.get("error", "") or "")


def build_preflight_env(
    *,
    env: dict[str, str] | None = None,
    isaaclab_root: str | Path = DEFAULT_ISAACLAB_ROOT,
) -> dict[str, str]:
    child_env = dict(os.environ if env is None else env)
    additions = [str(MODULE_ROOT)]
    root = Path(isaaclab_root)
    source = root / "source"
    if source.exists():
        for extension_dir in source.iterdir():
            if extension_dir.is_dir():
                additions.append(str(extension_dir))
    existing = child_env.get("PYTHONPATH", "")
    pieces = additions + ([existing] if existing else [])
    child_env["PYTHONPATH"] = os.pathsep.join(piece for piece in pieces if piece)
    return child_env


def detect_eula_prompt(text: str | bytes | None) -> bool:
    if text is None:
        return False
    if isinstance(text, bytes):
        value = text.decode("utf-8", errors="replace")
    else:
        value = str(text)
    lower = value.lower()
    return any(needle in lower for needle in EULA_PROMPT_NEEDLES)


def quote_windows_cmd_arg(value: str | Path) -> str:
    text = str(value)
    if text == "":
        return '""'
    return '"' + text.replace('"', '""') + '"'


def _inspect_current_process(isaaclab_root: str | Path) -> IsaacInterpreterReport:
    started = time.monotonic()
    report = IsaacInterpreterReport(
        executable=sys.executable,
        python_version=sys.version.replace("\n", " "),
        environment=inspect_launcher_environment(),
    )
    _add_isaaclab_source_paths(isaaclab_root)
    try:
        import importlib.metadata as metadata

        try:
            report.isaacsim_version = metadata.version("isaacsim")
        except Exception:
            report.isaacsim_version = _read_version_file_from_env("ISAAC_PATH")
        try:
            report.isaaclab_version = metadata.version("isaaclab")
        except Exception:
            report.isaaclab_version = _read_isaaclab_version_file(isaaclab_root)

        try:
            import isaacsim  # noqa: F401

            report.isaacsim_importable = True
        except Exception:
            report.traceback = traceback_module.format_exc()
            raise

        try:
            import isaaclab  # noqa: F401

            report.isaaclab_importable = True
        except Exception:
            report.traceback = traceback_module.format_exc()
            raise

        try:
            from isaaclab.app import AppLauncher  # noqa: F401

            report.app_launcher_importable = True
        except Exception:
            report.traceback = traceback_module.format_exc()
            raise

        report.compatible_python, reason = validate_isaac_python_compatibility(
            report.python_version,
            report.isaacsim_version,
        )
        if not report.compatible_python:
            report.error = reason
            report.error_category = "python_version_mismatch"
    except Exception as exc:
        if not report.error:
            report.error = str(exc)
        if not report.traceback:
            report.traceback = traceback_module.format_exc()
        report.error_category = classify_startup_error(report.error + "\n" + report.traceback)
    report.elapsed_s = max(0.0, time.monotonic() - started)
    if not report.error_category:
        report.error_category = classify_startup_error(report.error or report.traceback)
    return report


def _add_isaaclab_source_paths(isaaclab_root: str | Path) -> None:
    root = Path(isaaclab_root)
    source = root / "source"
    if source.exists():
        for extension_dir in source.iterdir():
            if extension_dir.is_dir():
                path = str(extension_dir)
                if path not in sys.path:
                    sys.path.append(path)
    root_path = str(root)
    if root.exists() and root_path not in sys.path:
        sys.path.append(root_path)


def _read_version_file_from_env(env_name: str) -> str:
    root = os.environ.get(env_name, "")
    if not root:
        return ""
    path = Path(root) / "VERSION"
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _read_isaaclab_version_file(isaaclab_root: str | Path) -> str:
    try:
        return (Path(isaaclab_root) / "VERSION").read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _parse_python_major_minor(version_text: str) -> tuple[int, int]:
    import re

    match = re.search(r"(\d+)\.(\d+)", str(version_text or ""))
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _parse_major(version_text: str) -> int:
    import re

    match = re.search(r"(\d+)", str(version_text or ""))
    return int(match.group(1)) if match else 0


def _parse_report_from_output(stdout: str | bytes | None) -> IsaacInterpreterReport | None:
    if stdout is None:
        return None
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout)
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            data = json.loads(stripped)
            return IsaacInterpreterReport(**{key: data.get(key) for key in IsaacInterpreterReport.__dataclass_fields__})
        except Exception:
            continue
    return None


def _join_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    def normalize(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    return normalize(stdout) + ("\n" if stdout and stderr else "") + normalize(stderr)


def _child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-child", action="store_true")
    parser.add_argument("--isaaclab-root", type=str, default=str(DEFAULT_ISAACLAB_ROOT))
    args = parser.parse_args(argv)
    report = _inspect_current_process(args.isaaclab_root)
    print(json.dumps(report.to_dict(), ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0 if not report.error else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return _child_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
