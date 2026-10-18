"""Exact FLUX.3 Diffusers names mapped to the native checkpoint vocabulary."""


def native_flux3_tensor_name(name: str) -> str:
    """Translate one immutable Diffusers tensor name without changing its values."""
    roots = {
        "context_embedder.weight": "txt_in.weight",
        "x_embedders.image.weight": "emb_in.image.weight",
        "x_embedders.image_cond.weight": "emb_in.image_cond.weight",
        "norm_out.linear.weight": "final_layer.image.adaLN_modulation.1.weight",
        "proj_out.weight": "final_layer.image.linear.weight",
        "timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "target_timestep_embedder.linear_1.weight": "ref_time_in.in_layer.weight",
        "target_timestep_embedder.linear_2.weight": "ref_time_in.out_layer.weight",
    }
    if name in roots:
        return roots[name]
    for prefix in ("early_stream_modulations.", "single_stream_modulations."):
        if name.startswith(prefix):
            return name.replace(".linear.weight", ".lin.weight")
    for prefix, target, limit in (
        ("mode_blocks.image.", "content_mode_blocks.image.", 5),
        ("mode_blocks.image_cond.", "content_mode_blocks.image_cond.", 5),
        ("txt_mode_blocks.", "txt_mode_blocks.", 5),
        ("single_transformer_blocks.", "single_blocks.", 28),
    ):
        if not name.startswith(prefix):
            continue
        index, separator, suffix = name[len(prefix) :].partition(".")
        if not separator or not index.isdecimal() or not 0 <= int(index) < limit:
            raise ValueError("FLUX.3 Diffusers block index is outside its geometry")
        suffixes = {
            "attn.norm_k.weight": "norm.key_norm.scale",
            "attn.norm_q.weight": "norm.query_norm.scale",
            "attn.to_k.weight": "k_proj.weight",
            "attn.to_q.weight": "q_proj.weight",
            "attn.to_v.weight": "v_proj.weight",
            "attn.to_out.0.weight": "attn_out.weight",
            "ff.linear_in.weight": "mlp_in.weight",
            "ff.linear_out.weight": "mlp_out.weight",
        }
        if suffix in suffixes:
            return target + index + "." + suffixes[suffix]
    raise ValueError("FLUX.3 Diffusers tensor has no established native mapping")
