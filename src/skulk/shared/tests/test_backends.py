# pyright: reportPrivateUsage=false
"""Tests for the backend capability tag vocabulary.

Probing and derivation behavior (which tags a node advertises given what it
observed and declared) lives in ``skulk.facts`` and is tested with synthetic
facts in ``src/skulk/facts/tests/``; this module covers the tag vocabulary,
resolution, and the thin ``probe_node_backends`` delegation.
"""

import sys

import pytest

import skulk.facts
from skulk.facts.derive import BackendDerivation
from skulk.shared.backends import (
    ACE_STEP_COMPUTE_BACKENDS,
    ENGINE_PACKAGES_DISABLE_ENV,
    ENGINE_UNKNOWN_FAILURE_MARKER,
    backend_hardware_present,
    describe_backend_hardware,
    engine_of,
    engine_packages_disabled,
    engine_supports_multi_node,
    expand_engine_backends,
    make_backend_tag,
    package_engine_of,
    platform_compatible_backends,
    probe_node_backends,
    resolve_node_backend,
    resolve_node_engine,
    unknown_engine_refusal,
)


def test_make_backend_tag_is_compound() -> None:
    assert make_backend_tag("mlx", "metal") == "mlx-metal"
    assert make_backend_tag("mlx_audio", "metal") == "mlx_audio-metal"
    assert make_backend_tag("llama_cpp", "vulkan") == "llama_cpp-vulkan"
    assert make_backend_tag("ace_step", "rocm") == "ace_step-rocm"


def test_native_ace_vocabulary_does_not_invent_a_vulkan_implementation() -> None:
    """The native package/runtime contract has exactly four explicit device lanes."""
    assert engine_of("ace_step") == engine_of("ace_step-cuda") == "ace_step"
    assert expand_engine_backends(frozenset({"ace_step"})) == ACE_STEP_COMPUTE_BACKENDS | {"ace_step"}
    assert "ace_step-vulkan" not in ACE_STEP_COMPUTE_BACKENDS
    assert not engine_supports_multi_node("ace_step")
    assert platform_compatible_backends(
        frozenset({"ace_step", "audio_cpp", "mlx", "ace_step-vulkan"}),
        card_serves_vision=False, card_serves_music=True, card_has_native_music_profile=True,
    ) == ACE_STEP_COMPUTE_BACKENDS


@pytest.mark.parametrize("music", [False, True])
def test_native_ace_cannot_load_a_non_native_card(music: bool) -> None:
    """Broad engine declarations cannot send GGUF music or text to native ACE."""
    assert not platform_compatible_backends(
        expand_engine_backends(frozenset({"ace_step"})), card_serves_vision=False, card_serves_music=music,
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("mlx", "mlx"),
        ("mlx-metal", "mlx"),
        ("mlx_audio", "mlx_audio"),
        ("mlx_audio-metal", "mlx_audio"),
        ("llama_cpp", "llama_cpp"),
        ("llama_cpp-vulkan", "llama_cpp"),
        ("llama_cpp-rocm", "llama_cpp"),
        ("cuda", None),  # bare compute, no engine
        ("vllm", "vllm"),
        ("vllm-cuda", "vllm"),
        ("bogus-cuda", None),  # unknown engine
        ("", None),
    ],
)
def test_engine_of(tag: str, expected: str | None) -> None:
    assert engine_of(tag) == expected


@pytest.mark.parametrize(
    ("backends", "expected"),
    [
        (frozenset({"mlx"}), "an Apple Silicon Mac"),
        (frozenset({"mlx", "mlx-metal"}), "an Apple Silicon Mac"),
        # mlx-audio advertises only when its package imports, so a Mac may
        # lack it; a speech card never claims a hardware requirement.
        (frozenset({"mlx_audio"}), None),
        (frozenset({"mlx", "mlx_audio"}), None),
        # vLLM and llama.cpp depend on installed software as well as hardware,
        # so a card that can use them never claims a hardware requirement.
        (frozenset({"mlx", "llama_server-cuda"}), None),
        (frozenset({"vllm-cuda"}), None),
        # The video engine installs on demand wherever its wheel sets apply.
        (
            frozenset({"comfy", "comfy-cuda", "comfy-rocm"}),
            "a Linux machine with an NVIDIA GPU or an AMD Strix Halo GPU",
        ),
        (frozenset({"bogus"}), None),
        (frozenset[str](), None),
    ],
)
def test_describe_backend_hardware(
    backends: frozenset[str], expected: str | None
) -> None:
    assert describe_backend_hardware(backends) == expected


def test_backend_hardware_present_recognizes_a_mac_without_mlx() -> None:
    assert backend_hardware_present(frozenset({"mlx"}), frozenset({"platform:darwin"}))
    assert not backend_hardware_present(
        frozenset({"mlx"}), frozenset({"platform:linux", "nvidia"})
    )
    assert not backend_hardware_present(
        frozenset({"llama_server-cuda"}), frozenset({"platform:darwin"})
    )


def test_video_engine_hardware_matches_the_rocm_wheel_set() -> None:
    from skulk.provisioning.manifest import COMFY_ROCM_PCI_DEVICE_IDS
    from skulk.shared.backends import _ENGINE_HARDWARE

    amd_classes = {
        hardware_class
        for hardware_class in _ENGINE_HARDWARE["comfy"][1]
        if hardware_class.startswith("amd:")
    }
    assert amd_classes == {
        f"amd:pci-{device.replace(':', '-')}" for device in COMFY_ROCM_PCI_DEVICE_IDS
    }
    assert backend_hardware_present(
        frozenset({"comfy-cuda"}), frozenset({"platform:linux", "nvidia"})
    )
    # Another AMD GPU cannot run the recorded ROCm wheels.
    assert not backend_hardware_present(
        frozenset({"comfy"}), frozenset({"platform:linux", "amd", "amd:pci-1002-744c"})
    )


def test_probe_includes_mlx_on_darwin() -> None:
    tags = probe_node_backends()
    if sys.platform == "darwin":
        # Bare engine tag kept for back-compat with original {"mlx"} cards.
        assert "mlx" in tags
        assert "mlx-metal" in tags
    else:
        assert "mlx" not in tags


def test_probe_node_backends_delegates_to_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The public probe entry point is a thin delegate over the process-wide
    # facts snapshot; consumers must see exactly the derived tags.
    monkeypatch.setattr(
        skulk.facts,
        "current_backend_derivation",
        lambda: BackendDerivation(backends=frozenset({"llama_server-cuda"})),
    )
    assert probe_node_backends() == frozenset({"llama_server-cuda"})


def test_resolve_node_engine_existing_mlx_cards_unchanged() -> None:
    # An original {"mlx"} card on a Mac node ({"mlx","mlx-metal"}) -> mlx.
    engine = resolve_node_engine(
        frozenset({"mlx"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine == "mlx"


def test_resolve_node_engine_picks_llama_cpp() -> None:
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cpu"}),
        ("llama_cpp-vulkan", "llama_cpp-rocm"),
        frozenset({"llama_cpp", "llama_cpp-vulkan"}),
    )
    assert engine == "llama_cpp"


def test_resolve_node_engine_none_when_no_intersection() -> None:
    # Node advertises only mlx but the card requires llama_cpp -> no match
    # (placement would have excluded this node; caller falls back to default).
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine is None


def test_resolve_node_backend_returns_preferred_tag() -> None:
    # The winning tag (not just the engine) honors backend_preference order.
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cpu"}),
        ("llama_cpp-rocm", "llama_cpp-vulkan"),
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm"}),
    )
    assert tag == "llama_cpp-rocm"


def test_resolve_node_backend_falls_back_to_sorted_when_no_preference() -> None:
    # With no preference match, the intersection is ordered deterministically,
    # and a CPU compute tag never beats a GPU tag on alphabetical accident:
    # GPU serving dominates CPU for every model class shipped, so the platform
    # default puts -cpu last (a card can still prefer CPU explicitly).
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
        (),
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
    )
    assert tag == "llama_cpp-vulkan"
    # The served-engine shape from the card sweep: no llama_server preference,
    # node advertises GPU + CPU server builds -> GPU wins, never -ngl 0.
    tag = resolve_node_backend(
        frozenset({"llama_server-vulkan", "llama_server-cpu"}),
        ("llama_cpp-vulkan",),
        frozenset({"llama_server-vulkan", "llama_server-cpu"}),
    )
    assert tag == "llama_server-vulkan"
    # An explicit CPU preference still wins: the fallback order is a platform
    # default, not an override of card policy.
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
        ("llama_cpp-cpu",),
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
    )
    assert tag == "llama_cpp-cpu"


def test_platform_compatible_backends_gates_vision_off_served() -> None:
    # MODEL truth vs PLATFORM truth: a vision card keeps every declared tag on
    # engines whose runner can load its projector (in-process llama.cpp, MLX)
    # and loses the served llama_server tags until that runner passes mmproj.
    declared = frozenset(
        {
            "mlx",
            "llama_cpp-vulkan",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cpu",
        }
    )
    assert platform_compatible_backends(declared, card_serves_vision=True) == (
        frozenset({"mlx", "llama_cpp-vulkan", "llama_cpp-cpu"})
    )
    # A text-only card is untouched: no platform gate applies.
    assert (
        platform_compatible_backends(declared, card_serves_vision=False) == declared
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=True,
        card_has_pinned_projector=True,
    ) == declared


def test_vision_card_preferring_llama_server_still_resolves_to_llama_cpp() -> None:
    # A vision GGUF card ships with a served-first backend_preference (MODEL
    # truth: the model's artifacts run on llama_server too). On a node that
    # advertises BOTH engines, the platform filter must still strip the served
    # tags before resolution so the card lands on the in-process llama.cpp runner
    # that can load its mmproj projector. This guards the card sweep that made
    # every GGUF card prefer llama_server: the vision exception is owned entirely
    # by code, so the served preference on a vision card is a safe no-op.
    declared = frozenset(
        {
            "llama_cpp-vulkan",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cpu",
        }
    )
    served_first = (
        "llama_server-vulkan",
        "llama_server-cpu",
        "llama_cpp-vulkan",
        "llama_cpp-cpu",
    )
    node_backends = frozenset({"llama_server-vulkan", "llama_cpp-vulkan"})
    platform_backends = platform_compatible_backends(
        declared, card_serves_vision=True
    )
    assert "llama_server-vulkan" not in platform_backends
    tag = resolve_node_backend(platform_backends, served_first, node_backends)
    assert tag == "llama_cpp-vulkan"
    # A text card on the same node honors the served preference (no gate applies).
    text_backends = platform_compatible_backends(declared, card_serves_vision=False)
    assert (
        resolve_node_backend(text_backends, served_first, node_backends)
        == "llama_server-vulkan"
    )


def test_platform_compatible_backends_gates_speech_to_mlx_audio() -> None:
    # Speech cards stay on the dedicated speech engine until another runner owns
    # the TTS/STT contract.
    declared = frozenset(
        {
            "mlx",
            "mlx-metal",
            "mlx_audio",
            "mlx_audio-metal",
            "llama_cpp-vulkan",
            "future_speech_engine",
        }
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_serves_speech=True,
    ) == frozenset(
        {"mlx_audio", "mlx_audio-metal", "future_speech_engine"}
    )


def test_platform_compatible_backends_gates_embeddings_to_their_runner() -> None:
    # Embedding cards run on engines with an embedding runner: the transformers
    # runner (mlx placements) and a dedicated llama-server --embeddings. A card
    # declaring another engine must be refused at placement rather than crash at
    # runner import on a node without that runner.
    declared = frozenset(
        {
            "mlx",
            "mlx-metal",
            "llama_server-vulkan",
            "llama_server-cuda",
            "llama_cpp-vulkan",
            "vllm-cuda",
            "future_engine",
        }
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_serves_embeddings=True,
    ) == frozenset(
        {
            "mlx",
            "mlx-metal",
            "llama_server-vulkan",
            "llama_server-cuda",
            "future_engine",
        }
    )
    # Non-embedding cards are untouched by the gate.
    assert (
        platform_compatible_backends(declared, card_serves_vision=False) == declared
    )


def test_metal_llama_server_lane_serves_only_embeddings() -> None:
    # MLX owns chat on the Apple GPU; the Metal llama-server lane exists for
    # GGUF embedding cards, so a GGUF chat card never lands on it.
    declared = frozenset({"llama_server-metal", "llama_server-vulkan", "llama_server"})
    assert platform_compatible_backends(
        declared, card_serves_vision=False, card_serves_embeddings=True
    ) == declared
    assert platform_compatible_backends(declared, card_serves_vision=False) == (
        frozenset({"llama_server-vulkan", "llama_server"})
    )


def test_a_bare_engine_tag_never_implies_the_embedding_only_lane() -> None:
    expanded = expand_engine_backends(frozenset({"llama_server"}))
    assert "llama_server-metal" not in expanded
    assert {"llama_server", "llama_server-vulkan", "llama_server-cpu"} <= expanded
    # A card that names the lane keeps it.
    assert "llama_server-metal" in expand_engine_backends(
        frozenset({"llama_server-metal"})
    )


@pytest.mark.parametrize(
    "declared",
    [
        # The worker's fallback check passes the card's tags unexpanded...
        frozenset({"llama_server"}),
        # ...and the master expands them first, keeping the bare tag.
        expand_engine_backends(frozenset({"llama_server"})),
    ],
)
def test_a_bare_llama_server_chat_card_never_resolves_onto_a_mac(
    declared: frozenset[str],
) -> None:
    mac = frozenset({"mlx", "mlx-metal", "llama_server-metal"})
    linux = frozenset({"llama_server", "llama_server-vulkan", "llama_server-cpu"})
    chat = platform_compatible_backends(declared, card_serves_vision=False)
    assert resolve_node_backend(chat, (), mac) is None
    assert resolve_node_backend(chat, (), linux) == "llama_server-vulkan"
    embedding = platform_compatible_backends(
        declared, card_serves_vision=False, card_serves_embeddings=True
    )
    assert resolve_node_backend(embedding, (), mac) == "llama_server-metal"
    assert resolve_node_backend(embedding, (), linux) == "llama_server-vulkan"


def test_platform_compatible_backends_requires_vllm_tool_parser() -> None:
    """Tool-using cards cannot land on a vLLM server without its parser pair."""
    declared = frozenset({"mlx", "vllm-cuda", "vllm-rocm"})

    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=True,
    ) == frozenset({"mlx"})
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=True,
        card_vllm_tool_call_parser="qwen3_xml",
    ) == declared
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=False,
    ) == declared


def test_resolve_node_backend_none_when_no_intersection() -> None:
    assert (
        resolve_node_backend(
            frozenset({"llama_cpp-vulkan"}), (), frozenset({"mlx", "mlx-metal"})
        )
        is None
    )


def test_resolve_node_engine_matches_backend_engine() -> None:
    # resolve_node_engine is exactly engine_of(resolve_node_backend(...)).
    compatible = frozenset({"llama_cpp-vulkan", "mlx"})
    preference = ("llama_cpp-vulkan",)
    node = frozenset({"llama_cpp-vulkan", "mlx"})
    tag = resolve_node_backend(compatible, preference, node)
    assert tag == "llama_cpp-vulkan"
    assert resolve_node_engine(compatible, preference, node) == engine_of(tag)


def test_engine_supports_multi_node() -> None:
    # MLX shards across nodes (ring/jaccl); llama.cpp is single-node until its
    # RPC runner lands (#328). This is the placement single-node guard's hinge.
    assert engine_supports_multi_node("mlx") is True
    assert engine_supports_multi_node("mlx_audio") is False
    assert engine_supports_multi_node("llama_cpp") is False


def test_platform_compatible_backends_gates_new_families_off_in_process_llama_cpp() -> None:
    """A family the pinned binding predates keeps only its served llama.cpp lanes."""
    declared = frozenset(
        {
            "llama_cpp-vulkan",
            "llama_cpp-cuda",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cuda",
            "mlx",
        }
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_family_predates_in_process_binding=True,
    ) == frozenset({"llama_server-vulkan", "llama_server-cuda", "mlx"})
    assert (
        platform_compatible_backends(declared, card_serves_vision=False) == declared
    )


@pytest.mark.parametrize(
    "tag", ["tensorfold-rocm", "tensorfold", "bogus-cuda", "cuda", "", "MLX-metal"]
)
def test_a_stamped_backend_of_an_unknown_engine_is_refused(tag: str) -> None:
    refusal = unknown_engine_refusal(tag)

    assert refusal is not None
    assert refusal.startswith(f"{ENGINE_UNKNOWN_FAILURE_MARKER}: ")
    assert repr(tag) in refusal


@pytest.mark.parametrize(
    "tag", [None, "mlx", "mlx-metal", "llama_server-vulkan", "vllm-cuda", "demucs-cpu"]
)
def test_a_known_or_unstamped_backend_is_not_refused(tag: str | None) -> None:
    # No stamped backend means the node resolves the engine from its own
    # probe, which is a separate path and not a refusal.
    assert unknown_engine_refusal(tag) is None


def _order_before_packages(
    compatible: frozenset[str], preference: tuple[str, ...], node: frozenset[str]
) -> str | None:
    """The choice made before engine packages existed, for sets without one."""
    usable = expand_engine_backends(compatible) & node
    usable = frozenset(
        tag
        for tag in usable
        if "-" in tag or not any(other.startswith(f"{tag}-") for other in usable)
    )
    if not usable:
        return None
    by_spelling = sorted(usable, key=lambda tag: (tag.endswith("-cpu"), tag))
    for wanted in preference:
        for tag in by_spelling:
            if tag == wanted or ("-" not in wanted and tag.startswith(f"{wanted}-")):
                return tag
    return by_spelling[0]


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        # Spelling alone would pick the package ahead of vLLM.
        (frozenset({"vllm-cuda", "tensorfold-cuda"}), "vllm-cuda"),
        (frozenset({"llama_server-vulkan", "tensorfold-rocm"}), "llama_server-vulkan"),
        # A built-in CPU engine is what served the model before the package came.
        (frozenset({"llama_cpp-cpu", "tensorfold-rocm"}), "llama_cpp-cpu"),
        # The only compatible engine the node has is still used.
        (frozenset({"tensorfold-rocm"}), "tensorfold-rocm"),
        (frozenset({"tensorfold-cpu", "tensorfold-rocm"}), "tensorfold-rocm"),
    ],
)
def test_with_no_preference_a_built_in_engine_is_chosen_before_a_package(
    node: frozenset[str], expected: str
) -> None:
    compatible = frozenset({"vllm", "llama_server", "llama_cpp", "tensorfold"})

    assert resolve_node_backend(compatible, (), node) == expected


@pytest.mark.parametrize("preference", [("tensorfold-rocm",), ("tensorfold",)])
def test_a_preference_that_names_a_package_chooses_it(
    preference: tuple[str, ...],
) -> None:
    compatible = frozenset({"llama_server", "tensorfold"})
    node = frozenset({"llama_server-vulkan", "tensorfold-rocm"})

    assert resolve_node_backend(compatible, preference, node) == "tensorfold-rocm"


@pytest.mark.parametrize(
    "node",
    [
        frozenset({"mlx", "mlx-metal"}),
        frozenset({"llama_cpp", "llama_cpp-cpu", "llama_cpp-vulkan", "llama_cpp-rocm"}),
        frozenset({"llama_server", "llama_server-cpu", "llama_server-cuda", "vllm", "vllm-cuda"}),
        frozenset({"llama_cpp-cpu", "llama_server-vulkan"}),
        frozenset({"vllm-rocm", "llama_server-rocm", "llama_cpp-vulkan"}),
        frozenset[str](),
    ],
)
@pytest.mark.parametrize(
    "preference",
    [(), ("llama_server-vulkan", "llama_cpp-rocm"), ("vllm",), ("llama_cpp",)],
)
def test_without_a_package_the_choice_is_what_it_was(
    node: frozenset[str], preference: tuple[str, ...]
) -> None:
    compatible = frozenset({"mlx", "llama_cpp", "llama_server", "vllm"})

    assert resolve_node_backend(compatible, preference, node) == _order_before_packages(
        compatible, preference, node
    )


def test_the_packages_switch_is_off_unless_set_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENGINE_PACKAGES_DISABLE_ENV, raising=False)
    assert not engine_packages_disabled()
    for value in ("", "0", "true", "yes"):
        monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, value)
        assert not engine_packages_disabled()
    for value in ("1", " 1 "):
        monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, value)
        assert engine_packages_disabled()


def test_a_refusal_says_when_packages_are_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    refusal = unknown_engine_refusal("tensorfold-rocm")
    bounded = unknown_engine_refusal("x" * 5000)

    assert refusal is not None
    assert refusal.startswith(f"{ENGINE_UNKNOWN_FAILURE_MARKER}: ")
    assert f"{ENGINE_PACKAGES_DISABLE_ENV}=1" in refusal
    assert bounded is not None and len(bounded) < 300
    # A built-in engine is never refused, whatever the switch says.
    assert unknown_engine_refusal("mlx-metal") is None


def test_an_unknown_engine_refusal_bounds_the_tag_it_repeats() -> None:
    refusal = unknown_engine_refusal("x" * 5000)

    assert refusal is not None and len(refusal) < 300


@pytest.mark.parametrize(
    ("tag", "name"),
    [
        ("tensorfold-rocm", "tensorfold"),
        ("tensorfold", "tensorfold"),
        ("tensor_fold2-cpu", "tensor_fold2"),
    ],
)
def test_a_package_shaped_tag_names_its_engine(tag: str, name: str) -> None:
    assert package_engine_of(tag) == name


@pytest.mark.parametrize(
    "tag",
    ["mlx", "mlx-metal", "llama_server-cuda", "cuda", "rocm-cuda", "", "MLX-metal",
     "tensor-fold-rocm", "tensor fold", "tensorfold-", "-rocm", "tensorfold-gpu"],
)
def test_built_in_reserved_and_malformed_tags_are_not_package_tags(tag: str) -> None:
    assert package_engine_of(tag) is None


def test_a_bare_package_engine_accepts_whichever_lane_a_node_holds() -> None:
    expanded = expand_engine_backends(frozenset({"tensorfold"}))

    assert expanded == frozenset(
        {"tensorfold", "tensorfold-metal", "tensorfold-vulkan", "tensorfold-rocm",
         "tensorfold-cuda", "tensorfold-cpu"}
    )
    # A named lane stays as narrow as the card wrote it.
    assert expand_engine_backends(frozenset({"tensorfold-rocm"})) == frozenset(
        {"tensorfold-rocm"}
    )


def test_a_card_naming_a_package_engine_resolves_to_the_nodes_lane() -> None:
    node = frozenset({"tensorfold-rocm", "llama_server-vulkan", "llama_server"})

    assert resolve_node_backend(frozenset({"tensorfold"}), (), node) == "tensorfold-rocm"
    # Preference by bare engine name works for a package as for a built-in.
    both = frozenset({"tensorfold", "llama_server"})
    assert resolve_node_backend(both, ("tensorfold",), node) == "tensorfold-rocm"
    assert resolve_node_backend(both, ("llama_server",), node) == "llama_server-vulkan"
    # A node without the package is unaffected by the card naming it.
    assert (
        resolve_node_backend(both, ("tensorfold",), frozenset({"llama_server-vulkan"}))
        == "llama_server-vulkan"
    )
    assert resolve_node_backend(frozenset({"tensorfold"}), (), frozenset({"mlx"})) is None


def test_a_bare_package_alias_never_wins_over_its_lane() -> None:
    node = frozenset({"tensorfold", "tensorfold-cuda"})

    assert resolve_node_backend(frozenset({"tensorfold"}), (), node) == "tensorfold-cuda"
