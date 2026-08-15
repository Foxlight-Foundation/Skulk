"""Product identity contract for the intelligent-fabric cognition."""

from skulk.api.steward import STEWARD_SYSTEM_PROMPT


def test_intelligent_fabric_prompt_identifies_as_skulk_itself() -> None:
    """The internal steward role must never become a separate character."""
    prompt = " ".join(STEWARD_SYSTEM_PROMPT.split())
    assert 'When asked your name, answer "Skulk."' in prompt
    assert "speak in the first person" in prompt
    assert "not a separate assistant, steward, or" in prompt
    assert "intelligent distributed AI fabric" in prompt
