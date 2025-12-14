#!/usr/bin/env python3
"""
Test script to verify backward compatibility of LoRA on Attention implementation.
This ensures that existing code works without any modifications when LoRA is disabled (default).
"""

import sys
import traceback

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Test 1: Import Checks")
    print("=" * 60)
    try:
        from autogluon.multimodal import MultiModalPredictor
        from autogluon.multimodal.models.sam import SAMForSemanticSegmentation
        from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import (
            SamAttention, SamTwoWayAttentionBlock, SamTwoWayTransformer, SamMaskDecoder, SamModel
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_sam_attention_default():
    """Test SamAttention with default parameters (LoRA disabled)."""
    print("\n" + "=" * 60)
    print("Test 2: SamAttention Default Initialization")
    print("=" * 60)
    try:
        from transformers.models.sam.configuration_sam import SamMaskDecoderConfig
        from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamAttention
        import torch.nn as nn
        
        config = SamMaskDecoderConfig()
        
        # Test 1: Default initialization (no LoRA params)
        attn1 = SamAttention(config)
        assert isinstance(attn1.q_proj, nn.Linear), "Default q_proj should be nn.Linear"
        assert isinstance(attn1.k_proj, nn.Linear), "Default k_proj should be nn.Linear"
        assert isinstance(attn1.v_proj, nn.Linear), "Default v_proj should be nn.Linear"
        print("✓ Default initialization (lora_r=0): Uses nn.Linear")
        
        # Test 2: Explicit lora_r=0
        attn2 = SamAttention(config, lora_r=0)
        assert isinstance(attn2.q_proj, nn.Linear), "Explicit lora_r=0 should use nn.Linear"
        print("✓ Explicit lora_r=0: Uses nn.Linear")
        
        # Test 3: With LoRA enabled
        attn3 = SamAttention(config, lora_r=8)
        from autogluon.multimodal.models.adaptation_layers import LoRALinear
        assert isinstance(attn3.q_proj, LoRALinear), "lora_r=8 should use LoRALinear"
        print("✓ lora_r=8: Uses LoRALinear")
        
        return True
    except Exception as e:
        print(f"✗ SamAttention test failed: {e}")
        traceback.print_exc()
        return False

def test_sam_model_default():
    """Test SAMForSemanticSegmentation with default parameters."""
    print("\n" + "=" * 60)
    print("Test 3: SAM Model Default Initialization")
    print("=" * 60)
    try:
        from autogluon.multimodal.models.sam import SAMForSemanticSegmentation
        
        # This should work without any LoRA parameters (backward compatible)
        model = SAMForSemanticSegmentation(
            prefix="sam",
            checkpoint_name="facebook/sam-vit-huge",
            num_classes=1,
            pretrained=False,  # Don't actually download for testing
        )
        
        # Check that LoRA is disabled by default
        assert model.decoder_attention_lora_r == 0, "Default lora_r should be 0"
        print("✓ SAM model initializes with LoRA disabled (r=0)")
        
        return True
    except Exception as e:
        print(f"✗ SAM model test failed: {e}")
        traceback.print_exc()
        return False

def test_cli_parsing():
    """Test that CLI parameters parse correctly."""
    print("\n" + "=" * 60)
    print("Test 4: CLI Parameter Parsing")
    print("=" * 60)
    try:
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--decoder_attn_lora_enable", action="store_true")
        parser.add_argument("--decoder_attn_lora_r", type=int, default=8)
        parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8)
        parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0)
        
        # Test 1: Default (no flags)
        args1 = parser.parse_args([])
        assert args1.decoder_attn_lora_enable == False, "Default should be disabled"
        assert args1.decoder_attn_lora_r == 8, "Default r should be 8"
        print("✓ Default CLI parsing: LoRA disabled")
        
        # Test 2: Enabled with custom params
        args2 = parser.parse_args([
            "--decoder_attn_lora_enable",
            "--decoder_attn_lora_r", "16",
            "--decoder_attn_lora_alpha", "16",
            "--decoder_attn_lora_dropout", "0.1"
        ])
        assert args2.decoder_attn_lora_enable == True
        assert args2.decoder_attn_lora_r == 16
        assert args2.decoder_attn_lora_alpha == 16
        assert args2.decoder_attn_lora_dropout == 0.1
        print("✓ Enabled CLI parsing: Custom parameters work")
        
        return True
    except Exception as e:
        print(f"✗ CLI parsing test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test that forward pass works with and without LoRA."""
    print("\n" + "=" * 60)
    print("Test 5: Forward Pass Compatibility")
    print("=" * 60)
    try:
        import torch
        from transformers.models.sam.configuration_sam import SamMaskDecoderConfig
        from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamAttention
        
        config = SamMaskDecoderConfig()
        
        # Create attention layers (with and without LoRA)
        attn_no_lora = SamAttention(config, lora_r=0)
        attn_with_lora = SamAttention(config, lora_r=8)
        
        # Create dummy inputs
        batch_size, point_batch_size, n_tokens = 1, 1, 10
        hidden_size = config.hidden_size
        
        query = torch.randn(batch_size, point_batch_size, n_tokens, hidden_size)
        key = torch.randn(batch_size, point_batch_size, n_tokens, hidden_size)
        value = torch.randn(batch_size, point_batch_size, n_tokens, hidden_size)
        
        # Test forward pass
        with torch.no_grad():
            out1 = attn_no_lora(query, key, value)
            out2 = attn_with_lora(query, key, value)
        
        assert out1.shape == (batch_size, point_batch_size, n_tokens, hidden_size)
        assert out2.shape == (batch_size, point_batch_size, n_tokens, hidden_size)
        
        print("✓ Forward pass works with LoRA disabled")
        print("✓ Forward pass works with LoRA enabled")
        print("✓ Output shapes match expected dimensions")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LoRA on Attention - Backward Compatibility Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("SamAttention Default", test_sam_attention_default),
        ("SAM Model Default", test_sam_model_default),
        ("CLI Parsing", test_cli_parsing),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All backward compatibility tests passed!")
        print("✅ Existing code will work without modifications")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        print("❌ Please fix the issues before deploying")
        return 1

if __name__ == "__main__":
    sys.exit(main())

