import torch
from transformers.models.sam.configuration_sam import SamMaskDecoderConfig

from autogluon.multimodal.models.adaptation_layers import GatedLoRALinear, LoRALinear
from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamAttention


def test_decoder_attention_lora_default_linear_type():
    cfg = SamMaskDecoderConfig()
    cfg.hidden_size = 256
    cfg.num_attention_heads = 8
    cfg.attention_downsample_rate = 2
    cfg.decoder_attention_lora_gate_enabled = False
    cfg.decoder_attention_lora_gate_noise_std = 0.0

    attn = SamAttention(cfg, downsample_rate=1, lora_r=8, lora_alpha=8, lora_dropout=0.0)
    assert isinstance(attn.q_proj, LoRALinear)
    assert not isinstance(attn.q_proj, GatedLoRALinear)


def test_decoder_attention_lora_gated_linear_type_and_gspo_mode():
    cfg = SamMaskDecoderConfig()
    cfg.hidden_size = 256
    cfg.num_attention_heads = 8
    cfg.attention_downsample_rate = 2
    cfg.decoder_attention_lora_gate_enabled = True
    cfg.decoder_attention_lora_gate_noise_std = 0.5

    attn = SamAttention(cfg, downsample_rate=1, lora_r=8, lora_alpha=8, lora_dropout=0.0)
    assert isinstance(attn.q_proj, GatedLoRALinear)

    # Toggle GSPO mode (should not error)
    attn.q_proj.set_gspo_mode(active=True, gate_noise_std=0.1, delta_noise_std=0.02)
    attn.q_proj.set_gspo_mode(active=False)

    # Forward shape sanity check
    q = torch.randn(2, 1, 16, 256)  # [B, point_batch, tokens, C]
    out = attn.q_proj(q)
    assert out.shape == (2, 1, 16, cfg.hidden_size)


