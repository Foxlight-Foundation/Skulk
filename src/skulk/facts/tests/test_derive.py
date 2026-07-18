# pyright: reportPrivateUsage=false
"""Derivation matrix: synthetic facts -> advertised backends + conflicts.

Every case here is the pure-function form of a real node shape: a fresh CUDA
pod with no env (#609), an NVIDIA box without nvidia-ml-py (#612), a typo'd
binary override (#462), a Strix Halo with a Vulkan llama-server, a Mac, and
the declared-override-vs-observed-hardware disagreements in between.
"""

from skulk.facts.derive import derive_node_backends
from skulk.shared.types.node_facts import (
    CapabilityConflictCode,
    EngineBinaryFact,
    GpuDeviceFact,
    LlamaServerDeviceProbe,
    NodeFacts,
)

NVIDIA_A40 = GpuDeviceFact(
    vendor="nvidia",
    name="NVIDIA A40",
    detection_source="nvml",
    vram_total_bytes=48 * 2**30,
    compute_capability="8.6",
)
NVIDIA_PRESENCE_ONLY = GpuDeviceFact(
    vendor="nvidia", detection_source="nvidia_device_node"
)
AMD_STRIX = GpuDeviceFact(
    vendor="amd",
    name="AMD GPU",
    detection_source="amdgpu_sysfs",
    vram_total_bytes=8 * 2**30,
    gtt_total_bytes=120 * 2**30,
)


def make_facts(
    *,
    platform: str = "linux",
    gpus: tuple[GpuDeviceFact, ...] = (),
    llama_cpp_importable: bool = False,
    llama_cpp_gpu_offload: bool | None = None,
    mlx_audio_importable: bool = False,
    llama_server_bin: EngineBinaryFact | None = None,
    vllm_bin: EngineBinaryFact | None = None,
    rpc_bin: EngineBinaryFact | None = None,
    device_probe: LlamaServerDeviceProbe | None = None,
    declared_llama_cpp: str | None = None,
    declared_llama_server: str | None = None,
    declared_vllm: str | None = None,
) -> NodeFacts:
    """Build a synthetic facts record with unconfigured-binary defaults."""
    return NodeFacts(
        platform=platform,
        gpus=gpus,
        llama_cpp_importable=llama_cpp_importable,
        llama_cpp_gpu_offload=llama_cpp_gpu_offload,
        mlx_audio_importable=mlx_audio_importable,
        llama_server_binary=llama_server_bin
        or EngineBinaryFact(env_var="SKULK_LLAMA_SERVER_BIN"),
        vllm_binary=vllm_bin or EngineBinaryFact(env_var="SKULK_VLLM_BIN"),
        rpc_server_binary=rpc_bin or EngineBinaryFact(env_var="SKULK_RPC_SERVER_BIN"),
        llama_server_device_probe=device_probe or LlamaServerDeviceProbe(),
        declared_llama_cpp_backends=declared_llama_cpp,
        declared_llama_server_backends=declared_llama_server,
        declared_vllm_backends=declared_vllm,
    )


def ok_bin(env_var: str) -> EngineBinaryFact:
    return EngineBinaryFact(env_var=env_var, configured_path=f"/opt/{env_var}", state="ok")


def bad_bin(env_var: str) -> EngineBinaryFact:
    return EngineBinaryFact(
        env_var=env_var, configured_path=f"/opt/{env_var}", state="missing"
    )


def conflict_codes(facts: NodeFacts) -> list[CapabilityConflictCode]:
    return [c.code for c in derive_node_backends(facts).conflicts]


# --- Darwin ---------------------------------------------------------------


def test_darwin_advertises_mlx() -> None:
    derivation = derive_node_backends(
        make_facts(
            platform="darwin",
            gpus=(GpuDeviceFact(vendor="apple", detection_source="apple_platform"),),
        )
    )
    assert derivation.backends == frozenset({"mlx", "mlx-metal"})
    assert derivation.conflicts == ()


def test_darwin_advertises_mlx_audio_when_importable() -> None:
    derivation = derive_node_backends(
        make_facts(
            platform="darwin",
            gpus=(GpuDeviceFact(vendor="apple", detection_source="apple_platform"),),
            mlx_audio_importable=True,
        )
    )
    assert {"mlx_audio", "mlx_audio-metal"} <= derivation.backends


def test_darwin_llama_cpp_stays_cpu_without_conflict() -> None:
    # An Apple GPU is not a "serving GPU" for the non-MLX engines: MLX owns
    # Metal, so llama_cpp-cpu on a Mac must never raise gpu_serving_disabled.
    derivation = derive_node_backends(
        make_facts(
            platform="darwin",
            gpus=(GpuDeviceFact(vendor="apple", detection_source="apple_platform"),),
            llama_cpp_importable=True,
            llama_cpp_gpu_offload=True,
        )
    )
    assert "llama_cpp-cpu" in derivation.backends
    assert derivation.conflicts == ()


# --- Bare nodes -----------------------------------------------------------


def test_bare_linux_node_advertises_nothing() -> None:
    derivation = derive_node_backends(make_facts())
    assert derivation.backends == frozenset()
    assert derivation.conflicts == ()


def test_cpu_only_linux_node_no_conflicts() -> None:
    # No GPU anywhere: the CPU floor is correct, not a degradation.
    derivation = derive_node_backends(
        make_facts(llama_cpp_importable=True, llama_cpp_gpu_offload=False)
    )
    assert derivation.backends == frozenset({"llama_cpp", "llama_cpp-cpu"})
    assert derivation.conflicts == ()


# --- llama_cpp declarations ----------------------------------------------


def test_llama_cpp_declared_backends_honored() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_cpp_importable=True,
            llama_cpp_gpu_offload=True,
            declared_llama_cpp="vulkan, rocm , bogus, metal",
        )
    )
    assert "llama_cpp-vulkan" in derivation.backends
    assert "llama_cpp-rocm" in derivation.backends
    assert "llama_cpp-bogus" not in derivation.backends
    assert "llama_cpp-metal" not in derivation.backends  # metal is MLX-only
    assert derivation.conflicts == ()


def test_llama_cpp_declared_gpu_dropped_when_build_is_cpu_only() -> None:
    # The wheel-clobber case: declared vulkan but the build reports no GPU
    # offload; drop to cpu AND raise the conflict so it is visible.
    facts = make_facts(
        gpus=(AMD_STRIX,),
        llama_cpp_importable=True,
        llama_cpp_gpu_offload=False,
        declared_llama_cpp="vulkan, rocm",
    )
    derivation = derive_node_backends(facts)
    assert "llama_cpp-vulkan" not in derivation.backends
    assert "llama_cpp-cpu" in derivation.backends
    codes = conflict_codes(facts)
    assert "backend_override_conflict" in codes
    # The whole-node CPU-only-with-GPU-visible signal also fires: nothing on
    # this node will use the GPU.
    assert "gpu_serving_disabled" in codes


def test_llama_cpp_declared_gpu_trusted_when_unverifiable() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_cpp_importable=True,
            llama_cpp_gpu_offload=None,
            declared_llama_cpp="vulkan",
        )
    )
    assert "llama_cpp-vulkan" in derivation.backends
    assert derivation.conflicts == ()
    assert any("could not verify" in note for note in derivation.notes)


def test_llama_cpp_declared_cuda_without_nvidia_is_loud_but_honored() -> None:
    facts = make_facts(
        llama_cpp_importable=True,
        llama_cpp_gpu_offload=True,
        declared_llama_cpp="cuda",
    )
    derivation = derive_node_backends(facts)
    # Configuration overrides detection; disagreement is loud.
    assert "llama_cpp-cuda" in derivation.backends
    assert conflict_codes(facts) == ["backend_override_conflict"]


# --- llama_cpp derivation (no declaration) --------------------------------


def test_llama_cpp_derives_cuda_from_nvidia_gpu() -> None:
    # The inversion: GPU offload verified + NVIDIA observed -> cuda, no env.
    derivation = derive_node_backends(
        make_facts(
            gpus=(NVIDIA_A40,),
            llama_cpp_importable=True,
            llama_cpp_gpu_offload=True,
        )
    )
    assert "llama_cpp-cuda" in derivation.backends
    assert derivation.conflicts == ()


def test_llama_cpp_derives_vulkan_and_rocm_from_amd_gpu() -> None:
    # An AMD build could be Vulkan or ROCm; the binding runs its compiled
    # backend regardless of tag, so advertise both and let cards prefer.
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_cpp_importable=True,
            llama_cpp_gpu_offload=True,
        )
    )
    assert {"llama_cpp-vulkan", "llama_cpp-rocm"} <= derivation.backends
    assert derivation.conflicts == ()


def test_llama_cpp_cpu_wheel_on_gpu_node_is_loud() -> None:
    # CPU-only wheel, GPU visible, no declaration, no other engine: the node
    # would silently serve everything on CPU -- the #609 class for llama_cpp.
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_cpp_importable=True,
        llama_cpp_gpu_offload=False,
    )
    derivation = derive_node_backends(facts)
    assert derivation.backends == frozenset({"llama_cpp", "llama_cpp-cpu"})
    assert conflict_codes(facts) == ["gpu_serving_disabled"]


# --- llama_server ---------------------------------------------------------


def test_served_not_configured_advertises_nothing() -> None:
    derivation = derive_node_backends(make_facts(gpus=(NVIDIA_A40,)))
    assert not any(tag.startswith("llama_server") for tag in derivation.backends)


def test_served_invalid_binary_is_loud() -> None:
    # #462 class: an explicit override that cannot be used must not read like
    # the env var was never set.
    facts = make_facts(llama_server_bin=bad_bin("SKULK_LLAMA_SERVER_BIN"))
    derivation = derive_node_backends(facts)
    assert not any(tag.startswith("llama_server") for tag in derivation.backends)
    assert conflict_codes(facts) == ["invalid_engine_binary"]


def test_served_declared_backends_honored() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
            declared_llama_server="vulkan, rocm , metal",
        )
    )
    assert {"llama_server", "llama_server-vulkan", "llama_server-rocm"} <= (
        derivation.backends
    )
    assert "llama_server-metal" not in derivation.backends
    assert derivation.conflicts == ()


def test_served_falls_back_to_llama_cpp_declaration() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
            declared_llama_cpp="vulkan",
        )
    )
    assert "llama_server-vulkan" in derivation.backends
    assert derivation.conflicts == ()


def test_served_derives_from_binary_device_list() -> None:
    # The binary's own --list-devices report is ground truth when no
    # declaration answers: a CUDA device list means cuda, no env needed (#609).
    derivation = derive_node_backends(
        make_facts(
            gpus=(NVIDIA_A40,),
            llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
            device_probe=LlamaServerDeviceProbe(
                outcome="devices", computes=("cuda",)
            ),
        )
    )
    assert "llama_server-cuda" in derivation.backends
    assert "llama_server-cpu" not in derivation.backends
    assert derivation.conflicts == ()


def test_served_derives_from_nvidia_hardware_when_probe_unsupported() -> None:
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        device_probe=LlamaServerDeviceProbe(outcome="unsupported"),
    )
    derivation = derive_node_backends(facts)
    assert "llama_server-cuda" in derivation.backends
    assert derivation.conflicts == ()


def test_served_derives_vulkan_from_amd_hardware() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
            device_probe=LlamaServerDeviceProbe(outcome="failed", detail="boom"),
        )
    )
    assert "llama_server-vulkan" in derivation.backends
    assert derivation.conflicts == ()


def test_served_cpu_build_on_gpu_node_is_loud() -> None:
    # The measured #609 case: GPU visible, binary reports no GPU devices,
    # would launch -ngl 0 and crawl. Never silent.
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        device_probe=LlamaServerDeviceProbe(outcome="devices", computes=()),
    )
    derivation = derive_node_backends(facts)
    assert "llama_server-cpu" in derivation.backends
    assert conflict_codes(facts) == ["gpu_serving_disabled"]
    conflict = derive_node_backends(facts).conflicts[0]
    assert "CPU-only build" in conflict.message


def test_served_declared_cpu_on_gpu_node_is_loud() -> None:
    # Even an explicit cpu declaration on a GPU node is a loud disagreement:
    # the operator may not know what they left on the table.
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        declared_llama_server="cpu",
    )
    assert conflict_codes(facts) == ["gpu_serving_disabled"]


def test_served_cpu_floor_without_gpu_is_silent() -> None:
    facts = make_facts(llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"))
    derivation = derive_node_backends(facts)
    assert "llama_server-cpu" in derivation.backends
    assert derivation.conflicts == ()


def test_served_declared_cuda_without_nvidia_is_loud_but_honored() -> None:
    facts = make_facts(
        gpus=(AMD_STRIX,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        declared_llama_server="cuda",
    )
    derivation = derive_node_backends(facts)
    assert "llama_server-cuda" in derivation.backends
    assert conflict_codes(facts) == ["backend_override_conflict"]


# --- vllm -----------------------------------------------------------------


def test_vllm_invalid_binary_is_loud() -> None:
    facts = make_facts(vllm_bin=bad_bin("SKULK_VLLM_BIN"))
    assert conflict_codes(facts) == ["invalid_engine_binary"]


def test_vllm_declared_backends_honored_gpu_only() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(NVIDIA_A40,),
            vllm_bin=ok_bin("SKULK_VLLM_BIN"),
            declared_vllm="cuda, vulkan , metal, cpu",
        )
    )
    assert derivation.backends == frozenset({"vllm", "vllm-cuda"})


def test_vllm_derives_cuda_from_nvidia_hardware() -> None:
    derivation = derive_node_backends(
        make_facts(gpus=(NVIDIA_A40,), vllm_bin=ok_bin("SKULK_VLLM_BIN"))
    )
    assert derivation.backends == frozenset({"vllm", "vllm-cuda"})
    assert derivation.conflicts == ()


def test_vllm_derives_rocm_from_amd_hardware() -> None:
    derivation = derive_node_backends(
        make_facts(gpus=(AMD_STRIX,), vllm_bin=ok_bin("SKULK_VLLM_BIN"))
    )
    assert derivation.backends == frozenset({"vllm", "vllm-rocm"})


def test_vllm_on_gpuless_node_is_loud_and_disabled() -> None:
    facts = make_facts(vllm_bin=ok_bin("SKULK_VLLM_BIN"))
    derivation = derive_node_backends(facts)
    assert not any(tag.startswith("vllm") for tag in derivation.backends)
    assert conflict_codes(facts) == ["backend_override_conflict"]


def test_vllm_falls_back_through_declaration_chain() -> None:
    derivation = derive_node_backends(
        make_facts(
            gpus=(AMD_STRIX,),
            vllm_bin=ok_bin("SKULK_VLLM_BIN"),
            declared_llama_cpp="rocm",
        )
    )
    assert derivation.backends == frozenset({"vllm", "vllm-rocm"})


# --- cross-cutting conflicts ---------------------------------------------


def test_nvidia_presence_without_nvml_is_degraded_detection() -> None:
    # #612: hardware visibly present, nvidia-ml-py absent -> loud warn with
    # the exact install remediation.
    facts = make_facts(
        gpus=(NVIDIA_PRESENCE_ONLY,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        declared_llama_server="cuda",
    )
    codes = conflict_codes(facts)
    assert "gpu_detection_degraded" in codes
    degraded = next(
        c
        for c in derive_node_backends(facts).conflicts
        if c.code == "gpu_detection_degraded"
    )
    assert "nvidia-ml-py" in degraded.remediation


def test_rpc_override_invalid_is_loud() -> None:
    # #462 verbatim: explicit SKULK_RPC_SERVER_BIN pointing nowhere.
    facts = make_facts(rpc_bin=bad_bin("SKULK_RPC_SERVER_BIN"))
    assert conflict_codes(facts) == ["invalid_engine_binary"]


def test_fresh_cuda_pod_with_bin_and_zero_backend_env_serves_gpu() -> None:
    # The Phase 0 exit shape (#614): a CUDA node with a llama-server binary
    # and ZERO SKULK_*_BACKENDS env derives GPU serving with no conflicts.
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_cpp_importable=False,
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        device_probe=LlamaServerDeviceProbe(outcome="devices", computes=("cuda",)),
    )
    derivation = derive_node_backends(facts)
    assert "llama_server-cuda" in derivation.backends
    assert derivation.conflicts == ()
