"""Verify native reference-audio token order, freezing and bounded admission."""

import pytest
import torch
from skulk_comfy_ltx.inpaint_audio import inpaint_audio_context


def test_reference_audio_keeps_the_complete_time_and_native_token_order() -> None:
    """Each token contains channels then frequencies for one actual VAE time."""
    source = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    before = source.clone()
    tokens, mask = inpaint_audio_context(source, maximum_bytes=65536, check_cancelled=lambda: None)
    assert tokens.shape == (1, 5, 6)
    for index in range(5):
        assert torch.equal(tokens[0, index], source[0, :, index].reshape(-1))
    assert mask.shape == (1, 1, 5, 1)
    assert mask.dtype == torch.float32 and mask.device.type == "cpu"
    assert bool((mask == 0).all()) and torch.equal(source, before)


@pytest.mark.parametrize("invalid", ["batch", "dimension", "nan", "budget"])
def test_reference_audio_refuses_malformed_or_unbudgeted_context(invalid: str) -> None:
    """Neither unknown batches nor an invalid source may become frozen context."""
    source = torch.zeros(2 if invalid == "batch" else 1, 2, 5, 3)
    if invalid == "dimension":
        source = source.squeeze(0)
    if invalid == "nan":
        source[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        inpaint_audio_context(source, maximum_bytes=1 if invalid == "budget" else 65536, check_cancelled=lambda: None)


def test_reference_audio_cancellation_precedes_token_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted request cannot retain another source token batch."""
    source = torch.zeros(1, 2, 5, 3)

    def cancel() -> None:
        raise InterruptedError("cancelled")

    def no_allocation(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("cancelled audio allocated a mask")

    monkeypatch.setattr(torch, "zeros", no_allocation)
    with pytest.raises(InterruptedError):
        inpaint_audio_context(source, maximum_bytes=65536, check_cancelled=cancel)
