"""skulk doctor: the executable node environment contract (#614).

The check registry lives in :mod:`skulk.doctor.checks`; the CLI in
:mod:`skulk.doctor.cli` (dispatched from ``skulk doctor``). User-facing
platform documentation is generated from the registry by
``scripts/generate_doctor_docs.py`` so docs and checks cannot drift apart.
"""

from skulk.doctor.checks import CheckResult, CheckVerdict, run_checks, run_fixes

__all__ = ["CheckResult", "CheckVerdict", "run_checks", "run_fixes"]
