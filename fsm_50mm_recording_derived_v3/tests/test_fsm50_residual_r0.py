from __future__ import annotations

import json
from pathlib import Path

from fsm_50mm_recording_derived_v3.fsm50_residual_r0 import (
    default_reviewed_runs,
    run_shadow_r0,
    validate_reviewed_macro_run_zero_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_each_reviewed_success_stream_has_exact_zero_identity() -> None:
    total = 0
    for spec in default_reviewed_runs(PROJECT_ROOT):
        summary, evidence = validate_reviewed_macro_run_zero_identity(spec)
        assert summary["passed"] is True
        assert summary["nominal_command_stream_sha256"] == summary["applied_command_stream_sha256"]
        assert summary["residual_dispatch_count"] == 0
        assert len(evidence) == spec.expected_actions
        assert all(row["zero_identity"] for row in evidence)
        assert all(row["nominal_targets_sha256"] == row["applied_targets_sha256"] for row in evidence)
        total += len(evidence)
    assert total == 112 + 119 + 132


def test_shadow_r0_writes_an_honestly_labeled_sha_bound_result(tmp_path: Path) -> None:
    result_path = run_shadow_r0(PROJECT_ROOT, tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["classification"] == "R0_SHADOW_ZERO_IDENTITY_PASS"
    assert result["live_isaac_execution"] is False
    assert result["source_action_count"] == 363
    assert result["zero_identity_count"] == 363
    assert result["residual_dispatch_count"] == 0
    assert Path(result["ledger_path"]).is_file()
