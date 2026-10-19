"""Resource advertisements retain each engine's separate first-use recipe."""

import pytest

import skulk.facts as facts_module
from skulk.facts.derive import BackendDerivation
from skulk.facts.testing import make_facts
from skulk.shared.types.profiling import NodeResources


async def test_gather_keeps_all_installable_builds_without_claiming_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging video support must preserve text/image preparation and readiness."""
    recipes = {
        "stable_diffusion_cpp-metal": "image-build",
        "vllm-cuda": "text-recipe",
        "mlx_video-metal": "video-build",
    }
    facts = make_facts().model_copy(
        update={
            "stable_diffusion_cpp_installable_builds": {"stable_diffusion_cpp-metal": recipes["stable_diffusion_cpp-metal"]},
            "vllm_installable_builds": {"vllm-cuda": recipes["vllm-cuda"]},
            "mlx_video_installable_builds": {"mlx_video-metal": recipes["mlx_video-metal"]},
        }
    )
    tags = frozenset(recipes)
    derivation = BackendDerivation(backends=tags, on_demand_backends=tags)
    monkeypatch.setattr(facts_module, "current_node_facts", lambda: facts)
    monkeypatch.setattr(facts_module, "current_backend_derivation", lambda: derivation)

    resources = await NodeResources.gather()

    assert resources.installable_engine_builds == recipes
    assert resources.on_demand_backends == tags
    assert resources.engine_builds == {}
