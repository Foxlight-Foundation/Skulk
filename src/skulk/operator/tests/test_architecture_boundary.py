"""Regression tests for the operator authority security-plane boundary."""

from pathlib import Path

import skulk.operator.authority as authority_module
import skulk.operator.identity as identity_module


def test_operator_persistence_does_not_import_inference_or_logging_planes() -> None:
    """Secrets cannot accidentally flow through State, events, telemetry, or logs."""

    forbidden_imports = (
        "skulk.shared.types.events",
        "skulk.shared.types.state",
        "skulk.shared.types.telemetry",
        "import logging",
        "from loguru",
    )
    module_paths = (
        Path(authority_module.__file__),
        Path(identity_module.__file__),
    )

    for module_path in module_paths:
        source = module_path.read_text(encoding="utf-8")
        for forbidden_import in forbidden_imports:
            assert forbidden_import not in source
