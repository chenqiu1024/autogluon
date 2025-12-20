
import torch
import torch.nn as nn
from transformers import SamConfig
from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamVisionLayer
from autogluon.multimodal.models.adaptation_layers import AdapterLayer

def test_sam_vision_layer_with_adapter():
    # 1. Setup Config
    config = SamConfig()
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.patch_size = 16
    config.image_size = 1024
    config.adapter_enabled = True
    config.adapter_dim = 32
    
    # 2. Instantiate Layer
    layer = SamVisionLayer(config, window_size=0)
    
    # 3. Verify Adapter Existence
    assert hasattr(layer, "adapter")
    assert isinstance(layer.adapter, AdapterLayer)
    assert layer.adapter.down_proj.out_features == 32
    
    # 4. Verify Forward Pass
    batch_size = 2
    seq_len = 16 * 16
    hidden_dim = 768
    x = torch.randn(batch_size, 16, 16, hidden_dim) # [B, H, W, C]
    
    output = layer(x)
    assert output[0].shape == (batch_size, 16, 16, hidden_dim)
    
    # 5. Verify Trainable Parameters
    adapter_params = list(layer.adapter.parameters())
    assert len(adapter_params) > 0
    for p in adapter_params:
        assert p.requires_grad

def test_sam_vision_layer_without_adapter():
    # 1. Setup Config
    config = SamConfig()
    config.hidden_size = 768
    config.adapter_enabled = False
    
    # 2. Instantiate Layer
    layer = SamVisionLayer(config, window_size=0)
    
    # 3. Verify Adapter Absence
    assert layer.adapter is None
    
    # 4. Verify Forward Pass
    batch_size = 2
    hidden_dim = 768
    x = torch.randn(batch_size, 16, 16, hidden_dim)
    
    output = layer(x)
    assert output[0].shape == (batch_size, 16, 16, hidden_dim)

if __name__ == "__main__":
    test_sam_vision_layer_with_adapter()
    test_sam_vision_layer_without_adapter()
    print("All tests passed!")
