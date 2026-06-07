"""Skulk — distributed MLX inference. Forked from exo; substantially diverged."""

import os

# Legacy environment compatibility (2026-06 exo -> skulk rename): alias any
# EXO_*-prefixed variable from a pre-rename deployment to its SKULK_* name.
# This runs at package import time — before any configuration is read, in
# every process including runner subprocesses — so operators' existing
# environments keep working unchanged. An explicit SKULK_* value always wins.
for _legacy_key, _legacy_value in list(os.environ.items()):
    if _legacy_key.startswith("EXO_"):
        os.environ.setdefault("SKULK_" + _legacy_key[len("EXO_") :], _legacy_value)
