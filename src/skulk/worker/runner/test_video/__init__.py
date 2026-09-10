"""Deterministic test video engine.

A hardware-free engine for the audio-video substrate: it renders a small
synthetic clip (moving shapes over a seeded gradient with a stereo tone),
muxes it into a minimal MP4, and drives every stage of the real pipeline:
card-declared duration and canvas rules, progress frames, the terminal
manifest, the output transfer, and job settlement. It exists so the
substrate is exercised end to end in tests and on nodes without a GPU. It
serves only the bundled ``foxlight/test-video`` card and is advertised when
``SKULK_TEST_VIDEO_ENGINE`` is set.
"""
