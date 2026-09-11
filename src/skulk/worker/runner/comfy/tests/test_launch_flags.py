"""The ComfyUI launch line varies only by the stamped compute backend."""

from __future__ import annotations

from skulk.worker.runner.comfy.runner import ROCM_LAUNCH_FLAGS, launch_flags


def test_rocm_backends_get_the_validated_strix_flags() -> None:
    assert launch_flags("comfy-rocm") == ROCM_LAUNCH_FLAGS
    assert ROCM_LAUNCH_FLAGS == ("--bf16-vae", "--disable-mmap", "--cache-none")


def test_other_backends_launch_with_the_headless_defaults_only() -> None:
    assert launch_flags("comfy-cuda") == ()
    assert launch_flags("comfy") == ()
    assert launch_flags(None) == ()
