from dataclasses import dataclass

@dataclass
class VideoInfo:
    width: int
    height: int
    num_frames: int
    fps: float
    duration: float
    has_audio: bool
