"""A plugin's own plain-language refusal, carried to the operator unchanged.

A managed owner may refuse an operation (turning a node on, saving settings,
asking for a burst offer) with a short sentence naming what to do next, such
as "Add your RunPod API key under Credentials". Core shows that sentence
instead of its own generic one. The text is author-written for operators, so
it is bounded and must be printable; anything else is dropped and the generic
sentence is shown, which keeps a refusal a refusal whatever the plugin sends.
"""

from typing import Final, final

MAX_REFUSAL_MESSAGE_CHARACTERS: Final = 300
"""The longest plugin refusal message core carries, after trimming."""


def refusal_message(value: object) -> str | None:
    """Return a plugin's refusal message when it is safe to show, else None.

    Args:
        value: The ``message`` member of an owner refusal frame, as decoded.

    Returns:
        The trimmed text when it is a string of 1 to 300 characters with no
        control or other unprintable characters, otherwise None.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not 1 <= len(text) <= MAX_REFUSAL_MESSAGE_CHARACTERS or not text.isprintable():
        return None
    return text


@final
class ManagedRefusalError(ValueError):
    """A managed owner refused an operation, optionally saying what to do next.

    It is a ``ValueError`` so every existing caller keeps treating it as an
    ordinary refusal (HTTP 409). ``message`` is the plugin's own next step for
    the operator. The constructor applies ``refusal_message`` itself, so an
    in-process provider raising this directly gets the same bounds as a
    managed owner's frame: an unusable message becomes None, and the caller's
    generic sentence applies.
    """

    def __init__(self, message: str | None) -> None:
        super().__init__("managed operation refused")
        self.message = refusal_message(message)


def refusal_detail(error: ValueError, fallback: str) -> str:
    """The HTTP detail for a refusal: the plugin's message, or the caller's own text.

    Args:
        error: The refusal raised while serving the request.
        fallback: The caller's generic sentence for a refusal without a message.

    Returns:
        The plugin's validated message when ``error`` carries one, else ``fallback``.
    """
    if isinstance(error, ManagedRefusalError) and error.message is not None:
        return error.message
    return fallback
