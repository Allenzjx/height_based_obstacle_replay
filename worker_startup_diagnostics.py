"""Startup diagnostics shared by the UI subprocess client and tests."""

from __future__ import annotations

from typing import Any


EULA_PROMPT_TOKENS = (
    "do you accept the eula",
    "accept the eula",
    "omni_kit_accept_eula",
    "omniverse eula",
)


def classify_startup_error(text: str) -> str:
    """Return a stable error category for launcher/worker startup text."""

    lower = str(text or "").lower()
    if '"" was unexpected at this time' in lower:
        return "batch_parse_error"
    if any(token in lower for token in EULA_PROMPT_TOKENS):
        return "eula_required"
    if "no module named 'isaacsim'" in lower or 'no module named "isaacsim"' in lower:
        return "missing_isaacsim"
    if "no module named 'isaaclab'" in lower or 'no module named "isaaclab"' in lower:
        return "missing_isaaclab"
    if "python version" in lower and ("mismatch" in lower or "requires" in lower or "compatible" in lower):
        return "python_version_mismatch"
    if "app launcher" in lower or "applauncher" in lower:
        return "app_launcher_error"
    if "sim.reset" in lower or "simulation reset" in lower:
        return "sim_reset_error"
    if "render" in lower or "renderer" in lower or "viewport" in lower:
        return "renderer_error"
    if "gpu" in lower or "driver" in lower or "cuda" in lower or "vulkan" in lower:
        return "gpu_or_driver_error"
    if "ipc" in lower or "connection refused" in lower or "timed out" in lower:
        return "ipc_connect_error"
    if "scene" in lower or "create_scene" in lower:
        return "scene_creation_error"
    if "timeout" in lower:
        return "startup_timeout"
    if "python" in lower and ("not found" in lower or "unable to find" in lower):
        return "missing_python"
    return "unknown"


def last_meaningful_line(lines: list[str] | tuple[str, ...] | str) -> str:
    """Pick the last non-empty, non-noise log line."""

    if isinstance(lines, str):
        source = lines.splitlines()
    else:
        source = [str(line) for line in lines]
    for line in reversed(source):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in {"[info]", "[warn]", "[warning]"}:
            continue
        return stripped
    return ""


def diagnose_startup_phase(
    *,
    phase: str,
    connected: bool,
    ready: bool,
    error_category: str = "",
    error: str = "",
    effective_headless: bool | None = None,
) -> str:
    """Produce a concise black-screen/startup next-step diagnosis."""

    phase = str(phase or "")
    category = str(error_category or classify_startup_error(error))
    if category and category != "unknown":
        return _diagnosis_for_category(category, error)
    if not connected and not ready:
        return "No IPC connection yet: check launcher, Windows batch quoting, Python interpreter, and preflight output."
    if phase in {"starting_app"}:
        return "Stopped while starting AppLauncher/SimulationApp: check EULA, extension download progress, experience, and HEADLESS/LIVESTREAM."
    if phase == "app_created":
        return "SimulationApp exists but SimulationContext was not created: inspect Isaac Lab SimulationContext and device configuration."
    if phase in {"creating_simulation_context", "simulation_context_created"}:
        return "Startup is inside SimulationContext creation: check Isaac Sim version, device, and GPU/driver logs."
    if phase in {"creating_robot", "robot_created", "creating_obstacle", "obstacle_created", "sim_reset_started"}:
        return "Scene exists but reset is not complete: inspect robot USD, obstacle creation, and sim.reset logs."
    if phase in {"adapter_ready", "running"} and effective_headless:
        return "Worker is running headless; a local viewport will remain unavailable unless GUI mode is restored."
    if phase in {"adapter_ready", "running"}:
        return "Worker is ready. If the viewport is black, check viewport camera, renderer warmup, GPU/driver logs, and livestream/headless settings."
    if phase:
        return f"Startup currently at phase '{phase}'. Check stdout/stderr tails for the next blocking detail."
    return "Startup has not reported a phase yet."


def _diagnosis_for_category(category: str, error: str) -> str:
    if category == "batch_parse_error":
        return 'Windows batch parsing failed. Use the config-file launcher path; the old long argv path can trigger "" was unexpected at this time.'
    if category == "missing_python":
        return "The requested Python executable was not found. Pick an explicit Isaac-compatible Python or repair the IsaacLab installation."
    if category == "missing_isaacsim":
        return "The selected Python cannot import isaacsim. Use auto mode or an explicit Isaac Sim Python instead of the active Conda Python."
    if category == "missing_isaaclab":
        return "The selected Python cannot import isaaclab. Install Isaac Lab into that interpreter or use the IsaacLab source launcher."
    if category == "python_version_mismatch":
        return "Python version is incompatible with the installed Isaac Sim package: Isaac Sim 5.x needs Python 3.11; Isaac Sim 4.x needs Python 3.10."
    if category == "eula_required":
        return "Isaac Sim requested EULA input. Run Isaac Sim once in the foreground or restart with --accept-isaac-eula if you explicitly agree."
    if category == "app_launcher_error":
        return "AppLauncher failed before SimulationApp became ready. Check IsaacLab version, experience path, and launcher environment."
    if category == "sim_reset_error":
        return "Simulation reset failed after scene creation. Check robot USD and physics setup logs."
    if category == "renderer_error":
        return "Renderer or viewport warmup failed. Check GUI/headless settings, GPU driver, and Isaac render logs."
    if category == "gpu_or_driver_error":
        return "GPU/driver initialization failed. Check NVIDIA driver, CUDA/Vulkan/RTX messages, and Isaac Sim compatibility."
    if category == "ipc_connect_error":
        return "The worker process did not establish IPC. Inspect stderr/stdout tails and the generated worker command."
    if category == "startup_timeout":
        return "Startup timed out without recent log activity. If this is the first Isaac run, increase --sim-startup-timeout-s and watch log growth."
    if category == "scene_creation_error":
        return "Scene creation failed. Check ground, lighting, robot USD, obstacle, and sim.reset phase history."
    return str(error or "Unknown startup failure.")


def summarize_environment(env: dict[str, Any] | None) -> dict[str, str]:
    """Keep only startup-relevant environment keys for UI/status display."""

    keys = (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "HEADLESS",
        "LIVESTREAM",
        "OMNI_KIT_ACCEPT_EULA",
        "ISAAC_PATH",
        "EXP_PATH",
        "CARB_APP_PATH",
    )
    source = env or {}
    return {key: str(source.get(key, "") or "") for key in keys}
