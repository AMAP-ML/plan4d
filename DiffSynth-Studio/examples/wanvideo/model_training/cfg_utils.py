import random

DEFAULT_CFG_SCALE = 4.0
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def validate_cfg_scale(value: float) -> float:
    if value < 1.0:
        raise ValueError(f"cfg_scale must be >= 1.0, got {value}.")
    return value


def resolve_negative_prompt(negative_prompt):
    return negative_prompt or ""


def compute_optimized_scale(positive_flat, negative_flat):
    dot_product = (positive_flat * negative_flat).sum(dim=1, keepdim=True)
    squared_norm = (negative_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def apply_cfg_zero_guidance(noise_pred_posi, noise_pred_nega, cfg_scale: float):
    cfg_scale = validate_cfg_scale(cfg_scale)
    positive = noise_pred_posi.reshape(noise_pred_posi.shape[0], -1)
    negative = noise_pred_nega.reshape(noise_pred_nega.shape[0], -1)
    st_star = compute_optimized_scale(positive, negative)
    st_star = st_star.view(noise_pred_nega.shape[0], *([1] * (noise_pred_nega.dim() - 1)))
    return noise_pred_nega * st_star + cfg_scale * (noise_pred_posi - noise_pred_nega * st_star)


def validate_dropout_probability(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def apply_prompt_dropout(prompt: str, dropout_prob: float) -> str:
    dropout_prob = validate_dropout_probability("prompt_dropout", dropout_prob)
    if dropout_prob <= 0.0 or prompt == "":
        return prompt
    if random.random() < dropout_prob:
        return ""
    return prompt
