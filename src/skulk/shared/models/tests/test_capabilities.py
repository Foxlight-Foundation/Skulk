

def test_is_nemotron_family() -> None:
    from skulk.shared.models.capabilities import is_nemotron_family
    from skulk.shared.models.model_cards import ModelId

    assert is_nemotron_family(
        model_id=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits")
    )
    assert is_nemotron_family(
        model_id=ModelId("mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit")
    )
    assert not is_nemotron_family(model_id=ModelId("mlx-community/Qwen3.5-9B-MLX-4bit"))
    assert not is_nemotron_family()
