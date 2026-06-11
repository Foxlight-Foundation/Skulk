"""Request-time context-length admission for the LLM runner (#145).

The within-request KV cache grows one entry per token with no upper bound, so
before this module existed a request whose prompt + output exceeded what the
hosting node(s) could hold crashed the runner mid-generation with an unhandled
Metal OOM (SIGABRT, wired GPU memory leaked). Admission converts that crash
into a deterministic, client-visible rejection *before* prefill starts.

The token limit itself (``instance_context_token_limit``) is computed once per
instance by the worker from static inputs and handed to the runner at spawn;
see ``skulk.shared.models.memory_estimate``. Every rank of a multi-node
instance computes the same verdict from the same inputs — divergent verdicts
would deadlock the ring collectives, which is why admission happens after
tokenization (identical on every rank) and never consults live memory.
"""

from typing import final

from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX


@final
class ContextLengthExceededError(Exception):
    """A request that cannot fit the instance's usable context window.

    The message is prefixed with ``CONTEXT_LENGTH_EXCEEDED_PREFIX`` so the API
    layer can recognize the rejection inside plain-string error fields and
    surface an OpenAI-style ``context_length_exceeded`` error instead of a 500.
    """

    def __init__(
        self,
        *,
        prompt_tokens: int,
        requested_max_output_tokens: int | None,
        context_token_limit: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.requested_max_output_tokens = requested_max_output_tokens
        self.context_token_limit = context_token_limit
        if requested_max_output_tokens is None:
            detail = (
                f"the prompt alone is {prompt_tokens} tokens, which leaves no "
                f"room for output within this instance's usable context limit "
                f"of {context_token_limit} tokens"
            )
        else:
            detail = (
                f"the prompt ({prompt_tokens} tokens) plus max_tokens "
                f"({requested_max_output_tokens}) exceeds this instance's "
                f"usable context limit of {context_token_limit} tokens"
            )
        super().__init__(
            f"{CONTEXT_LENGTH_EXCEEDED_PREFIX} {detail}. Reduce the prompt "
            f"length or max_tokens. The limit is the smaller of the model's "
            f"advertised context length and what fits in memory next to the "
            f"model weights on the hosting node(s)."
        )


def admit_max_tokens(
    *,
    prompt_tokens: int,
    requested_max_output_tokens: int | None,
    default_max_output_tokens: int,
    context_token_limit: int | None,
) -> int:
    """Resolve the output-token budget for a request, or reject it.

    Semantics follow the OpenAI API: a request whose prompt cannot fit (or
    whose *explicit* ``max_tokens`` cannot fit on top of the prompt) is
    rejected; an omitted ``max_tokens`` is clamped to the remaining context so
    generation ends with ``finish_reason="length"`` instead of overrunning.

    Args:
        prompt_tokens: Full tokenized prompt length, including any
            prefix-cache hit (the cached prefix still occupies KV memory).
        requested_max_output_tokens: The client's explicit ``max_tokens``,
            or ``None`` when omitted.
        default_max_output_tokens: Server default applied when the client
            omits ``max_tokens``.
        context_token_limit: The instance's usable context ceiling, or
            ``None`` when no ceiling is enforceable (preserves pre-#145
            behavior).

    Returns:
        The admitted output-token budget (always >= 1).

    Raises:
        ContextLengthExceededError: When the request cannot fit.
    """
    if context_token_limit is None:
        return (
            requested_max_output_tokens
            if requested_max_output_tokens is not None
            else default_max_output_tokens
        )
    if prompt_tokens >= context_token_limit:
        raise ContextLengthExceededError(
            prompt_tokens=prompt_tokens,
            requested_max_output_tokens=requested_max_output_tokens,
            context_token_limit=context_token_limit,
        )
    remaining = context_token_limit - prompt_tokens
    if requested_max_output_tokens is not None:
        if requested_max_output_tokens > remaining:
            raise ContextLengthExceededError(
                prompt_tokens=prompt_tokens,
                requested_max_output_tokens=requested_max_output_tokens,
                context_token_limit=context_token_limit,
            )
        return requested_max_output_tokens
    return min(default_max_output_tokens, remaining)
