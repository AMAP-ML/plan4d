import re


_BLOCK_KEY_PATTERN = re.compile(r"^encoder\.block\.(\d+)\.(.+)$")


def _rename_longcat_t5_key(name: str):
    if name == "shared.weight":
        return "token_embedding.weight"
    if name == "encoder.final_layer_norm.weight":
        return "norm.weight"

    match = _BLOCK_KEY_PATTERN.match(name)
    if match is None:
        return None
    block_id, suffix = match.groups()
    prefix = f"blocks.{block_id}."

    if suffix == "layer.0.layer_norm.weight":
        return prefix + "norm1.weight"
    if suffix == "layer.1.layer_norm.weight":
        return prefix + "norm2.weight"
    if suffix == "layer.0.SelfAttention.q.weight":
        return prefix + "attn.q.weight"
    if suffix == "layer.0.SelfAttention.k.weight":
        return prefix + "attn.k.weight"
    if suffix == "layer.0.SelfAttention.v.weight":
        return prefix + "attn.v.weight"
    if suffix == "layer.0.SelfAttention.o.weight":
        return prefix + "attn.o.weight"
    if suffix == "layer.0.SelfAttention.relative_attention_bias.weight":
        return prefix + "pos_embedding.embedding.weight"
    if suffix == "layer.1.DenseReluDense.wi_0.weight":
        return prefix + "ffn.gate.0.weight"
    if suffix == "layer.1.DenseReluDense.wi_1.weight":
        return prefix + "ffn.fc1.weight"
    if suffix == "layer.1.DenseReluDense.wo.weight":
        return prefix + "ffn.fc2.weight"
    return None


def WanVideoTextEncoderStateDictConverter(state_dict):
    # If keys are already in WanTextEncoder format, keep them unchanged.
    if "token_embedding.weight" in state_dict:
        return {name: state_dict[name] for name in state_dict}

    converted = {}
    for name in state_dict:
        mapped_name = _rename_longcat_t5_key(name)
        if mapped_name is not None:
            converted[mapped_name] = state_dict[name]
    return converted
