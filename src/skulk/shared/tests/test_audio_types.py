"""Invariant coverage for internal speech task parameter contracts."""

import pytest
from pydantic import ValidationError

from skulk.shared.types.audio import SpeechSynthesisTaskParams
from skulk.shared.types.common import ModelId


def test_bundled_reference_profile_rejects_request_scoped_conditioning() -> None:
    """A speech task must carry exactly one reference-conditioning source."""

    with pytest.raises(ValidationError, match="cannot be combined"):
        SpeechSynthesisTaskParams(
            model=ModelId("org/tts"),
            input_text="hello",
            reference_voice_profile="angus",
            reference_audio="/tmp/reference.wav",
        )
