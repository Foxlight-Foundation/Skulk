"""What the pinned engine-package chat dialect cannot carry.

Every engine package serves one chat dialect, and that dialect has no tool
calls, no images and no per-token log probabilities. Two places need to know:

- the API, which refuses such a request before it is dispatched. A refusal made
  by a runner arrives after the HTTP status is committed, so it reaches the
  caller as a 200 with an error object in the body, and a client that checks
  only the status takes it for an answer;
- the generic package runner, which stays the final guard for a request that
  reaches it another way.

Both read the functions here, so the two cannot disagree. The module imports no
runner code: the API must not load an engine to answer this.
"""

from collections.abc import Iterable

from skulk.shared.backends import package_engine_of
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import Instance


def unsupported_request_reason(params: TextGenerationTaskParams) -> str | None:
    """Say why a request is outside the pinned dialect, or return ``None``."""
    if params.tools:
        return (
            "This model is served by an engine package, whose chat dialect "
            "carries no tool calls. Retry without tools."
        )
    if params.images or params.image_count:
        return (
            "This model is served by an engine package, whose chat dialect "
            "carries no images. Retry with text only."
        )
    # OpenAI treats ``top_logprobs`` alone as a logprobs request, as the
    # built-in runners do.
    if params.logprobs or params.top_logprobs is not None:
        return (
            "Per-token logprobs are not supported by an engine package: its "
            "chat dialect does not carry them. Retry without logprobs."
        )
    return None


def served_only_by_engine_packages(instances: Iterable[Instance]) -> bool:
    """Return whether every given instance runs wholly on an engine package.

    False for no instances at all, and false as soon as one shard runs on a
    built-in engine or has no engine stamped yet: the request may then be
    served by an engine that can carry it, so it must not be refused up front.
    """
    seen = False
    for instance in instances:
        shards = instance.shard_assignments.runner_to_shard.values()
        if not shards:
            return False
        for shard in shards:
            backend = shard.resolved_backend
            if backend is None or package_engine_of(backend) is None:
                return False
        seen = True
    return seen


def engine_package_refusal(
    instances: Iterable[Instance], params: TextGenerationTaskParams
) -> str | None:
    """Say why ``params`` cannot be served when only engine packages could serve it.

    ``instances`` are the instances the request could be dispatched to. The
    answer is ``None`` when the request is inside the dialect, or when at
    least one of those instances runs on a built-in engine.
    """
    reason = unsupported_request_reason(params)
    if reason is None or not served_only_by_engine_packages(instances):
        return None
    return reason
