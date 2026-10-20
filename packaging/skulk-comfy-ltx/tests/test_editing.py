"""Preservation and causal-clock contracts without model weights or GPU rendering."""

import pytest
import torch
from skulk_comfy_ltx.editing import (
    AudioWave,
    edit_latent,
    normalize_source_audio,
    preserve_audio,
    preserve_video,
    temporal_mask,
)


def test_causal_masks_use_each_modality_clock() -> None:
    """Video's first frame and audio's first hop have different causal spans."""
    video = torch.zeros((2, 3, 7, 2, 3))
    mask = temporal_mask(video, modality="video", start_seconds=1 / 24, end_seconds=9 / 24, fps=24)
    assert torch.equal(mask.flatten(), torch.tensor([0, 1, 0, 0, 0, 0, 0]))
    audio = torch.zeros((2, 3, 7, 4))
    mask = temporal_mask(audio, modality="audio", start_seconds=0.01, end_seconds=0.05, fps=24)
    assert torch.equal(mask.flatten(), torch.tensor([0, 1, 0, 0, 0, 0, 0]))
    other_fps = temporal_mask(audio, modality="audio", start_seconds=0.01, end_seconds=0.05, fps=240)
    assert torch.equal(mask, other_fps)


def test_audio_retake_retains_the_encoder_tail_token() -> None:
    """Dropping the 52nd token would make a retake near the clip end a no-op."""
    source = torch.arange(52, dtype=torch.float32).reshape(1, 1, 52, 1)
    target = torch.zeros((1, 1, 51, 1))
    result, mask = edit_latent(source, target, modality="audio", operation="retake", start_seconds=2.02, end_seconds=2.04, fps=24)
    assert torch.equal(result, source) and result is not source
    assert mask.shape == source.shape
    assert torch.equal(mask.flatten(), torch.tensor([0] * 51 + [1]))


@pytest.mark.parametrize("audio", [True, False])
def test_extension_freezes_the_complete_encoded_prefix(audio: bool) -> None:
    """Both layouts preserve all source batches while leaving a fresh tail."""
    shape = (2, 3, 3, 4) if audio else (2, 3, 3, 2, 4)
    source = torch.full(shape, 0.25)
    target = torch.full((2, 3, 7, *shape[3:]), 0.75)
    result, mask = edit_latent(source, target, modality="audio" if audio else "video", operation="extend", start_seconds=0.0, end_seconds=0.0, fps=24)
    assert torch.equal(result[:, :, :3], source)
    assert torch.equal(result[:, :, 3:], target[:, :, 3:])
    assert torch.equal(mask.flatten(), torch.tensor([0, 0, 0, 1, 1, 1, 1]))
    assert bool((target == 0.75).all()) and bool((source == 0.25).all())


def test_audio_hold_freezes_every_token() -> None:
    """A video retake can use frozen audio as context without regenerating it."""
    source = torch.ones((1, 2, 20, 4))
    result, mask = edit_latent(source, torch.zeros_like(source), modality="audio", operation="retake", start_seconds=0.1, end_seconds=0.2, fps=24, regenerate_audio=False)
    assert torch.equal(result, source) and not bool(mask.any())


@pytest.mark.parametrize("extension", [True, False])
def test_original_frames_survive_decoder_drift(extension: bool) -> None:
    """Frozen latents alone cannot provide this decoded source-preservation contract."""
    frames = 9 if extension else 49
    source = torch.arange(frames * 2 * 3 * 3, dtype=torch.float32).reshape(frames, 2, 3, 3)
    generated = torch.full((17 if extension else 49, 2, 3, 3), -1.0)
    result = preserve_video(source, generated, operation="extend" if extension else "retake", fps=24, start_seconds=0.5, end_seconds=1.0)
    if extension:
        assert torch.equal(result[:9], source)
        assert bool((result[9:] == -1).all())
    else:
        assert torch.equal(result[:12], source[:12])
        assert torch.equal(result[24:], source[24:])
        assert bool((result[12:24] == -1).all())
    assert bool((generated == -1).all())


@pytest.mark.parametrize("regenerate", [True, False])
def test_original_pcm_survives_decoder_drift(regenerate: bool) -> None:
    """Preserve the requested source samples rather than VAE reconstructions."""
    source = AudioWave(waveform=torch.full((1, 2, 98000), 0.25), sample_rate=48000)
    generated = AudioWave(waveform=torch.full((1, 2, 100000), 0.75), sample_rate=48000)
    result = preserve_audio(source, generated, operation="retake", output_frames=49, fps=24, start_seconds=0.5, end_seconds=1.0, regenerate_audio=regenerate)
    assert result["waveform"].shape[-1] == 98000
    assert bool((result["waveform"][..., :24000] == 0.25).all())
    assert bool((result["waveform"][..., 48000:] == 0.25).all())
    assert bool((result["waveform"][..., 24000:48000] == (0.75 if regenerate else 0.25)).all())
    assert bool((generated["waveform"] == 0.75).all())


def test_extension_restores_source_pcm_before_new_audio() -> None:
    """The source prefix uses the audio sample clock rather than video token counts."""
    source = AudioWave(waveform=torch.full((1, 2, 50000), 0.25), sample_rate=48000)
    generated = AudioWave(waveform=torch.full((1, 2, 98000), 0.75), sample_rate=48000)
    result = preserve_audio(source, generated, operation="extend", output_frames=49, fps=24, start_seconds=0.0, end_seconds=0.0, regenerate_audio=True)
    assert torch.equal(result["waveform"][..., :50000], source["waveform"])
    assert bool((result["waveform"][..., 50000:] == 0.75).all())


def test_source_normalization_resamples_before_stereo_and_frame_alignment() -> None:
    """A non-output-rate mono source has a single explicit normalization step."""
    calls: list[tuple[int, int]] = []
    def resample(waveform: torch.Tensor, initial: int, target: int) -> torch.Tensor:
        calls.append((initial, target))
        return waveform.repeat_interleave(3, dim=-1)
    source = AudioWave(waveform=torch.full((1, 1, 6000), 0.125), sample_rate=16000)
    normalized = normalize_source_audio(source, frames=9, fps=24, sample_rate=48000, resample=resample)
    assert calls == [(16000, 48000)]
    assert normalized["waveform"].shape == (1, 2, 18000)
    assert bool((normalized["waveform"] == 0.125).all())


def test_a_missing_source_track_provides_silent_context() -> None:
    """A retake may generate its region while keeping the absent track silent elsewhere."""
    def unexpected(_waveform: torch.Tensor, _initial: int, _target: int) -> torch.Tensor:
        raise AssertionError("silence needs no resampler")
    source = normalize_source_audio(None, frames=9, fps=24, sample_rate=48000, resample=unexpected)
    assert source["waveform"].shape == (1, 2, 18000)
    assert not bool(source["waveform"].any())


def test_refuses_a_short_source_instead_of_extending_it_silently() -> None:
    """Only a bounded codec tail may be padded; missing content is an error."""
    def identity(waveform: torch.Tensor, _initial: int, _target: int) -> torch.Tensor:
        return waveform
    source = AudioWave(waveform=torch.zeros((1, 2, 100)), sample_rate=48000)
    with pytest.raises(ValueError, match="does not cover"):
        normalize_source_audio(source, frames=9, fps=24, sample_rate=48000, resample=identity)


def test_refuses_nonfinite_latents_and_changed_geometry() -> None:
    """Bad context cannot enter sampling or be silently resized."""
    source = torch.zeros((1, 2, 7, 2, 3))
    with pytest.raises(ValueError, match="geometry"):
        edit_latent(source, torch.zeros((1, 2, 7, 3, 3)), modality="video", operation="retake", start_seconds=0.1, end_seconds=0.2, fps=24)
    source[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        temporal_mask(source, modality="video", start_seconds=0.1, end_seconds=0.2, fps=24)
