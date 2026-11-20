"""Actual pinned constructor translation and real tensor result boundaries.

These tests import the curated upstream library but never initialize weights or
claim model generation. They catch API drift before expensive hardware runs.
"""

from __future__ import annotations

import importlib
import io
import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from pydantic import JsonValue, TypeAdapter
from skulk_ace_step.audio_separation import source_from_wav
from skulk_ace_step.bindings import soundfile
from skulk_ace_step.contracts import NativeMusicRequest
from skulk_ace_step.generation import prepare_call, result_samples
from skulk_ace_step.upstream_bindings import NativeHandler, native_constructors


class HandlerFactory(Protocol):
    """Construct the actual handler without loading its model components."""

    def __call__(self) -> NativeHandler:
        """Return the pinned task-instruction formatter."""
        ...


@dataclass
class TensorResult:
    """Controlled upstream completion envelope containing actual Torch tensors."""

    success: bool = True
    error: str | None = None
    audios: list[dict[str, object]] = field(default_factory=list)


def snapshot(value: object) -> dict[str, JsonValue]:
    """Inspect the actual native dataclass after its constructor/default handling."""
    return TypeAdapter(dict[str, JsonValue]).validate_json(
        json.dumps(vars(value), allow_nan=False)
    )


class NativeGenerationTests(unittest.TestCase):
    """No fake constructors, model loading, inference or server-local path reads."""

    def setUp(self) -> None:
        """Construct real pinned dataclasses and a weight-free native handler."""
        factory = cast(
            HandlerFactory,
            cast(object, importlib.import_module("acestep.handler").AceStepHandler),
        )
        self.handler = factory()
        self.constructors = native_constructors()
        self.base: dict[str, JsonValue] = {
            "request_id": "take-1",
            "operation": "generate",
            "prompt": "Acoustic folk, warm cello — a quiet return home.",
            "seed": 4294967295,
            "target_frames": 2 * 48000,
            "output_directory": "/private/native/take-1",
            "output_directory_identity": {"device": 1, "inode": 2},
        }
        stream = io.BytesIO()
        soundfile.write(
            stream,
            np.zeros((48000, 2), dtype=np.float32),
            48000,
            format="WAV",
            subtype="FLOAT",
        )
        self.source = source_from_wav(stream.getvalue())

    def request(self, **changes: JsonValue) -> NativeMusicRequest:
        """Validate a complete strict request after applying explicit test changes."""
        return NativeMusicRequest.model_validate_json(json.dumps(self.base | changes))

    def test_generation_retains_caption_lyrics_and_both_seed_paths(self) -> None:
        """Actual native defaults cannot replace the request or randomize its seed."""
        request = self.request(lyrics="[Verse]\nHome is a light I carry.")
        call = prepare_call(request, self.handler, self.constructors, None)
        params, config = snapshot(call.parameters), snapshot(call.configuration)
        self.assertEqual(params["caption"], request.prompt)
        self.assertEqual(params["lyrics"], request.lyrics)
        self.assertFalse(params["instrumental"])
        self.assertEqual(params["task_type"], "text2music")
        self.assertIsNone(params["src_audio"])
        self.assertEqual(params["duration"], 2.0)
        self.assertEqual(params["seed"], 4294967295)
        self.assertEqual(config["seeds"], [4294967295])
        self.assertEqual(config["batch_size"], 1)
        self.assertFalse(config["use_random_seed"])
        self.assertEqual(config["audio_format"], "wav32")
        for name in (
            "thinking",
            "use_cot_metas",
            "use_cot_caption",
            "use_cot_language",
            "dcw_enabled",
            "enable_normalization",
        ):
            self.assertFalse(params[name], name)
        self.assertEqual(
            (
                params["inference_steps"],
                params["guidance_scale"],
                params["repaint_strength"],
            ),
            (50, 7.0, 0.5),
        )

    def test_instrumental_marker_is_explicit_without_rewriting_caption(self) -> None:
        """Instrumental classification does not infer anything from a vocal prompt."""
        call = prepare_call(self.request(), self.handler, self.constructors, None)
        self.assertTrue(snapshot(call.parameters)["instrumental"])

    def test_editing_uses_canonical_context_native_clock_and_native_instruction(
        self,
    ) -> None:
        """LEGO selects its part; Continue repaints only the newly appended tail."""
        context = Path("/private/native/canonical.wav")
        for operation in ("add_part", "repaint", "continue"):
            with self.subTest(operation=operation):
                continuation = operation == "continue"
                request = self.request(
                    operation=operation,
                    source=self.source.model_dump(mode="json"),
                    source_path="/private/native/original.wav",
                    target_frames=96000 if continuation else 48000,
                    region_start_frame=48000 if continuation else 12000,
                    region_end_frame=96000 if continuation else 36000,
                    part_class="strings" if operation == "add_part" else None,
                )
                params = snapshot(
                    prepare_call(
                        request, self.handler, self.constructors, context
                    ).parameters
                )
                task = "lego" if operation == "add_part" else "repaint"
                self.assertEqual(params["task_type"], task)
                self.assertEqual(
                    params["instruction"],
                    self.handler.generate_instruction(
                        task, track_name=request.part_class
                    ),
                )
                if operation == "add_part":
                    self.assertIn("STRINGS", str(params["instruction"]))
                self.assertEqual(params["src_audio"], str(context))
                self.assertNotEqual(params["src_audio"], request.source_path)
                self.assertEqual(
                    params["repainting_start"], 1.0 if continuation else 0.25
                )
                self.assertEqual(
                    params["repainting_end"], 2.0 if continuation else 0.75
                )
                self.assertEqual(params["chunk_mask_mode"], "explicit")

    def test_conditioning_mismatch_is_refused_before_native_constructor(self) -> None:
        """A generation cannot accidentally become a source-conditioned edit."""
        with self.assertRaises(ValueError):
            prepare_call(
                self.request(),
                self.handler,
                self.constructors,
                Path("/private/context.wav"),
            )
        edit = self.request(
            operation="repaint",
            source=self.source.model_dump(mode="json"),
            source_path="/private/source.wav",
            target_frames=48000,
            region_end_frame=48000,
        )
        for context in (None, Path("relative.wav")):
            with self.subTest(context=context), self.assertRaises(ValueError):
                prepare_call(edit, self.handler, self.constructors, context)

    def test_actual_tensor_result_ignores_native_file_paths(self) -> None:
        """Output is measured from the native tensor, without opening its path."""
        tensor = torch.ones((2, 96000), dtype=torch.float32) * 0.25
        result = TensorResult(
            audios=[
                {"tensor": tensor, "sample_rate": 48000, "path": "/never/open/this.wav"}
            ]
        )
        rendered = result_samples(result, self.request())
        samples = rendered.samples
        self.assertEqual(rendered.padding_frames_removed, 0)
        self.assertEqual(samples.shape, (96000, 2))
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_array_equal(
            samples, np.full((96000, 2), 0.25, dtype=np.float32)
        )

    def test_wrong_tensor_geometry_precision_rate_and_finite_bounds_fail(self) -> None:
        """Reject plausible but incomplete/corrupt completions before WAV export."""
        cases: list[tuple[torch.Tensor, int]] = [
            (torch.zeros((2, 96000), dtype=torch.float16), 48000),
            (torch.zeros((1, 96000)), 48000),
            (torch.zeros((2, 47999)), 48000),
            (torch.zeros((2, 96000)), 44100),
            (torch.full((2, 96000), float("nan")), 48000),
            (torch.zeros((2, 1, 96000)), 48000),
        ]
        for tensor, rate in cases:
            with (
                self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype, rate=rate),
                self.assertRaises(ValueError),
            ):
                result_samples(
                    TensorResult(audios=[{"tensor": tensor, "sample_rate": rate}]),
                    self.request(),
                )
        edit = self.request(
            operation="repaint",
            source=self.source.model_dump(mode="json"),
            source_path="/private/source.wav",
            target_frames=48000,
            region_end_frame=48000,
        )
        with self.assertRaises(ValueError):
            result_samples(
                TensorResult(
                    audios=[{"tensor": torch.zeros((2, 96000)), "sample_rate": 48000}]
                ),
                edit,
            )
        for result in (
            TensorResult(success=False),
            TensorResult(),
            TensorResult(audios=[{}, {}]),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                result_samples(result, self.request())

    def test_short_edit_padding_cannot_hide_nonfinite_or_arbitrary_overruns(
        self,
    ) -> None:
        """Validate the full decoder tensor, even samples that would be removed."""
        edit = self.request(
            operation="repaint",
            source=self.source.model_dump(mode="json"),
            source_path="/private/source.wav",
            target_frames=48000,
            region_end_frame=48000,
        )
        for frames in (48001, 245759, 245761):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                result_samples(
                    TensorResult(
                        audios=[
                            {"tensor": torch.zeros((2, frames)), "sample_rate": 48000}
                        ]
                    ),
                    edit,
                )
        for invalid in (float("nan"), float("inf")):
            tensor = torch.zeros((2, 245760))
            tensor[0, -1] = invalid
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "nonfinite"),
            ):
                result_samples(
                    TensorResult(audios=[{"tensor": tensor, "sample_rate": 48000}]),
                    edit,
                )
        generated = result_samples(
            TensorResult(
                audios=[{"tensor": torch.zeros((2, 245760)), "sample_rate": 48000}]
            ),
            self.request(),
        )
        self.assertEqual(generated.samples.shape[0], 245760)
        self.assertEqual(generated.padding_frames_removed, 0)

    def test_long_edit_alignment_cannot_hide_short_nonfinite_or_arbitrary_output(
        self,
    ) -> None:
        """Explicit input alignment authorizes only its precise finite decoded tail."""
        edit = self.request(
            operation="continue",
            source=self.source.model_dump(mode="json"),
            source_path="/private/source.wav",
            target_frames=288001,
            region_start_frame=48000,
            region_end_frame=288001,
        )
        tensor = torch.zeros((2, 289920))
        rendered = result_samples(
            TensorResult(audios=[{"tensor": tensor, "sample_rate": 48000}]), edit
        )
        self.assertEqual(rendered.samples.shape, (288001, 2))
        self.assertEqual(rendered.padding_frames_removed, 1919)
        for frames in (288000, 288002, 289919, 289921):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                result_samples(
                    TensorResult(
                        audios=[
                            {"tensor": torch.zeros((2, frames)), "sample_rate": 48000}
                        ]
                    ),
                    edit,
                )
        for invalid in (float("nan"), float("inf")):
            tensor = torch.zeros((2, 289920))
            tensor[1, -1] = invalid
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "nonfinite"),
            ):
                result_samples(
                    TensorResult(audios=[{"tensor": tensor, "sample_rate": 48000}]),
                    edit,
                )
