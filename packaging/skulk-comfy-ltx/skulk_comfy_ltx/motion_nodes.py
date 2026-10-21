"""Expose only verified bounded trajectory files to the managed Comfy graph."""

from pathlib import Path
from typing import final

import comfy.model_management
import folder_paths
from comfy_api.latest import io

from skulk_comfy_ltx.motion_tracks import MAX_TRACK_BYTES, render_motion_tracks


@final
class SkulkLTXMotionTracks(io.ComfyNode):
    """Rasterize normalized trajectory media using the pinned trained format."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Describe a bounded input-file reference and the exact output geometry."""
        return io.Schema(node_id="SkulkLTXMotionTracks", display_name="Skulk LTX Motion Tracks", category="model/conditioning/ltxv", inputs=[io.String.Input("tracks_file", default="", multiline=False), io.Int.Input("width", default=512, min=128, max=4096), io.Int.Input("height", default=384, min=128, max=4096), io.Int.Input("expected_frames", default=25, min=9, max=2881)], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, tracks_file: str, width: int, height: int, expected_frames: int) -> io.NodeOutput:
        """Read only a regular bounded file beneath the request input directory."""
        root = Path(folder_paths.get_input_directory()).resolve()
        source = (root / tracks_file).resolve()
        if not source.is_relative_to(root) or not source.is_file() or source.stat().st_size > MAX_TRACK_BYTES:
            raise ValueError("Motion Track data must be a bounded file in the input directory")
        return io.NodeOutput(render_motion_tracks(source.read_bytes(), width, height, expected_frames, comfy.model_management.get_torch_device(), check_cancelled=comfy.model_management.throw_exception_if_processing_interrupted))
