import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal

import aiofiles
import aiofiles.os as aios
from loguru import logger

from skulk.shared.types.worker.shards import ShardMetadata


def filter_repo_objects[T](
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(p) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(p) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(
                f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string."
            )

        key = _identity

    for item in items:
        path = key(item)
        if allow_patterns is not None and not any(
            fnmatch(path, r) for r in allow_patterns
        ):
            continue
        if ignore_patterns is not None and any(
            fnmatch(path, r) for r in ignore_patterns
        ):
            continue
        yield item


def _add_wildcard_to_directories(pattern: str) -> str:
    if pattern[-1] == "/":
        return pattern + "*"
    return pattern


def get_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_home() -> Path:
    """Get the Hugging Face home directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


HfTokenSource = Literal["env", "service_env", "config", "file", "absent"]
"""Where a resolved Hugging Face token came from.

``env`` is the ``HF_TOKEN`` environment variable. ``service_env`` is
``HF_TOKEN`` in ``~/.skulk/skulk.env``, which the launchd and systemd wrappers
export into the service before it starts. ``config`` is ``hf_token:`` in
``skulk.yaml``, which node startup copies into ``HF_TOKEN`` when that variable
is unset. ``file`` is ``$HF_HOME/token`` (what ``hf auth login`` writes).
``absent`` means this node cannot authenticate to Hugging Face.

The order is the node's effective precedence. The service env file and
``hf_token`` both end up in ``HF_TOKEN`` before the token file is ever
consulted, and the wrapper's export happens before startup would apply
``hf_token``, so a standalone reader must consult them in this order to see
what the running service actually uses.
"""


def get_service_env_path() -> Path:
    """Return the service environment file the startup wrappers actually read.

    Fixed at ``~/.skulk/skulk.env``, which is what the systemd unit hardcodes
    (``EnvironmentFile=-%h/.skulk/skulk.env``) and what ``skulk-startup.sh``
    defaults to. Deliberately not derived from ``SKULK_HOME``: the wrappers do
    not consult it, so doing so would point doctor at a file the service never
    reads, which is the exact false report this source exists to prevent.

    The wrappers also honor a ``SKULK_ENV_FILE`` override, which this does not
    read, because that variable is shell-wrapper configuration and the runtime
    deliberately does not take new environment-driven configuration. On the
    rare node using that override, doctor reports the token as absent rather
    than reading a file it was not told about.
    """
    return Path.home() / ".skulk" / "skulk.env"


def _service_env_hf_token() -> str | None:
    """Return ``HF_TOKEN`` from ``~/.skulk/skulk.env``, or ``None``.

    The launchd and systemd wrappers source this file, so an installed service
    authenticates with whatever it holds. A standalone ``skulk doctor`` started
    from an operator shell does not run the wrapper, so without reading the
    file directly doctor would keep reporting a missing token right after the
    operator followed its own remediation.
    """
    try:
        contents = get_service_env_path().read_text()
    except OSError:
        return None
    # Last assignment wins, matching both consumers: the startup wrapper
    # sources this as shell, and systemd's EnvironmentFile also takes the final
    # value for a repeated key. Returning the first match would disagree with
    # the service whenever an operator appends a replacement token.
    resolved: str | None = None
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # `export HF_TOKEN=...` is valid in this file because skulk-startup.sh
        # sources it as shell, so the prefix must not hide the assignment.
        key = key.removeprefix("export ").strip()
        if key != "HF_TOKEN":
            continue
        # Shell-style quoting and trailing comments are both common here, and
        # the wrapper sources this as shell, where ` #` starts a comment.
        # Nothing in this function executes the file; it only unwraps what the
        # shell would have produced.
        value = re.split(r"\s+#", value, maxsplit=1)[0]
        resolved = value.strip().strip("\"'") or None
    return resolved


def get_hf_token_path() -> Path:
    """Return the file-based Hugging Face token path for this node.

    This is the location ``hf auth login`` writes and the only token source
    read per download rather than once at process start, which makes it the
    supported way to give a headless node a token without a restart.

    Delegates to ``huggingface_hub``'s own resolved constant rather than
    rebuilding it, so ``HF_TOKEN_PATH`` and the XDG cache location are honored
    exactly as the Hub CLI honors them. Recomputing it here as
    ``$HF_HOME/token`` silently missed both, which meant ``hf auth login`` could
    report success while writing somewhere Skulk never read.
    """
    try:
        from huggingface_hub import constants as hf_constants

        return Path(hf_constants.HF_TOKEN_PATH)
    except (ImportError, AttributeError):
        # Never let a Hub-internals change break token resolution outright;
        # fall back to the documented default layout.
        return get_hf_home() / "token"


def _config_hf_token() -> str | None:
    """Return ``hf_token:`` from ``skulk.yaml``, or ``None``.

    Read directly rather than via the environment because a standalone command
    such as ``skulk doctor`` runs before node startup would have promoted this
    value into ``HF_TOKEN``. Without it, the most common setup (a token saved
    through the dashboard, which persists here) reads as no token at all.
    """
    # Imported lazily: the download layer must not take a module-level
    # dependency on config loading.
    from skulk.store.config import load_skulk_config

    try:
        config = load_skulk_config()
    except Exception:  # noqa: BLE001 - a broken config is not this function's problem
        return None
    if config is None or not config.hf_token:
        return None
    return config.hf_token.strip() or None


def resolve_hf_token_source(
    *, include_config: bool = True
) -> tuple[str | None, HfTokenSource]:
    """Resolve this node's Hugging Face token synchronously, with provenance.

    Applies the node's effective precedence: ``HF_TOKEN``, then ``HF_TOKEN`` in
    the service env file, then ``hf_token:`` in ``skulk.yaml``, then
    ``$HF_HOME/token``. The two middle steps mirror what an installed node does
    before serving (the startup wrapper exports the env file; startup copies
    ``hf_token`` into ``HF_TOKEN`` when unset), so a synchronous caller sees the
    same token the running node would use.

    Callable from synchronous contexts such as ``skulk doctor``, and it reports
    which source won so operator-facing output can name the mechanism to change
    rather than merely asserting that a token exists.

    Args:
        include_config: When ``False``, skip the service env file and
            ``skulk.yaml`` and resolve exactly what an already-running process
            sees (``HF_TOKEN`` then the token file). Used for in-process
            callers and to pin agreement with :func:`get_hf_token`.

    Returns:
        A ``(token, source)`` pair. ``token`` is ``None`` exactly when
        ``source`` is ``"absent"``.
    """
    raw_environment_token = os.environ.get("HF_TOKEN")
    environment_token = (raw_environment_token or "").strip()
    if environment_token:
        return environment_token, "env"
    # A present-but-empty HF_TOKEN is not the same as an absent one. Node
    # startup only promotes hf_token when the key is *missing*, so a blank
    # export pins the node to the token file and the startup-only sources
    # below can never activate. Reporting one of them would claim a token the
    # downloader will not send.
    if include_config and raw_environment_token is None:
        if service_token := _service_env_hf_token():
            return service_token, "service_env"
        if config_token := _config_hf_token():
            return config_token, "config"
    token_path = get_hf_token_path()
    try:
        contents = token_path.read_text().strip()
    except OSError:
        # An unreadable or missing token file is simply "no token here": the
        # caller's job is to say so actionably, not to fail the audit.
        return None, "absent"
    if contents:
        return contents, "file"
    return None, "absent"


async def get_hf_token() -> str | None:
    """Retrieve the Hugging Face token from HF_TOKEN env var or HF_HOME directory.

    This is the in-process view: ``HF_TOKEN`` (which startup has already
    populated from ``hf_token:`` when it was unset) then ``$HF_HOME/token``. A
    regression test pins it to
    ``resolve_hf_token_source(include_config=False)`` so the operator guidance
    doctor prints cannot drift from what downloads actually use.
    """
    # Check environment variable first. Whitespace is not a credential: an
    # unstripped value here would be sent as a bearer token and rejected,
    # producing a confusing "token rejected" instead of falling through.
    if token := (os.environ.get("HF_TOKEN") or "").strip():
        return token
    # Fall back to file-based token
    token_path = get_hf_token_path()
    if await aios.path.exists(token_path):
        async with aiofiles.open(token_path, "r") as f:
            return (await f.read()).strip() or None
    return None


async def get_auth_headers() -> dict[str, str]:
    """Get authentication headers if a token is available."""
    token = await get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def extract_layer_num(tensor_name: str) -> int | None:
    # This is a simple example and might need to be adjusted based on the actual naming convention
    parts = tensor_name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def get_allow_patterns(weight_map: dict[str, str], shard: ShardMetadata) -> list[str]:
    default_patterns = set(
        [
            "*.json",
            "*.py",
            "tokenizer.model",
            "tiktoken.model",
            "*/spiece.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ]
    )
    shard_specific_patterns: set[str] = set()

    if shard.model_card.components is not None:
        shardable_component = next(
            (c for c in shard.model_card.components if c.can_shard), None
        )

        if weight_map and shardable_component:
            for tensor_name, filename in weight_map.items():
                # Strip component prefix from tensor name (added by weight map namespacing)
                # E.g., "transformer/blocks.0.weight" -> "blocks.0.weight"
                if "/" in tensor_name:
                    _, tensor_name_no_prefix = tensor_name.split("/", 1)
                else:
                    tensor_name_no_prefix = tensor_name

                # Determine which component this file belongs to from filename
                component_path = Path(filename).parts[0] if "/" in filename else None

                if component_path == shardable_component.component_path.rstrip("/"):
                    layer_num = extract_layer_num(tensor_name_no_prefix)
                    if (
                        layer_num is not None
                        and shard.start_layer <= layer_num < shard.end_layer
                    ):
                        shard_specific_patterns.add(filename)

                    if shard.is_first_layer or shard.is_last_layer:
                        shard_specific_patterns.add(filename)
                else:
                    shard_specific_patterns.add(filename)

        else:
            shard_specific_patterns = set(["*.safetensors"])

        # TODO(ciaran): temporary - Include all files from non-shardable components that have no index file
        for component in shard.model_card.components:
            if not component.can_shard and component.safetensors_index_filename is None:
                component_pattern = f"{component.component_path.rstrip('/')}/*"
                shard_specific_patterns.add(component_pattern)
    else:
        if weight_map:
            for tensor_name, filename in weight_map.items():
                layer_num = extract_layer_num(tensor_name)
                if (
                    layer_num is not None
                    and shard.start_layer <= layer_num < shard.end_layer
                ):
                    shard_specific_patterns.add(filename)
            layer_independent_files = set(
                [v for k, v in weight_map.items() if extract_layer_num(k) is None]
            )
            shard_specific_patterns.update(layer_independent_files)
            logger.debug(f"get_allow_patterns {shard=} {layer_independent_files=}")
        else:
            shard_specific_patterns = set(["*.safetensors"])

    logger.info(f"get_allow_patterns {shard=} {shard_specific_patterns=}")
    return list(default_patterns | shard_specific_patterns)
