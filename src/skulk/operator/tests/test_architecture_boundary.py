"""Regression tests for the operator authority security-plane boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import skulk.operator.authority as authority_module
import skulk.operator.identity as identity_module
import skulk.operator.replication as replication_module


def test_operator_persistence_does_not_import_inference_or_logging_planes() -> None:
    """Secrets cannot accidentally flow through State, events, telemetry, or logs."""

    forbidden_imports = (
        "skulk.shared.types.events",
        "skulk.shared.types.state",
        "skulk.shared.types.telemetry",
        "logging",
        "loguru",
    )
    module_paths = (
        Path(authority_module.__file__),
        Path(identity_module.__file__),
        Path(replication_module.__file__),
    )

    for module_path in module_paths:
        syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)
        for imported_module in imported_modules:
            assert not any(
                imported_module == forbidden_import
                or imported_module.startswith(f"{forbidden_import}.")
                for forbidden_import in forbidden_imports
            )
