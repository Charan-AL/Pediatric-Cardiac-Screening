import torch
from models.crnn_heart_sound import crnn_without_head
from models.nts_net_ultrasound import ntsnet_without_head

print("=" * 80)
print("DEBUGGING MODEL CHECKPOINT LOADING")
print("=" * 80)

# Test AUDIO model
print("\n1. AUDIO MODEL (CRNN2D)")
print("-" * 80)
try:
    audio_model = crnn_without_head("checkpoints/audio/audio_best.pth", device="cuda")
    print("✓ Audio model loaded successfully")
    
    # Test inference
    test_audio = torch.randn(1, 1, 64, 256).cuda()
    with torch.no_grad():
        audio_embed = audio_model(test_audio)
    print(f"✓ Audio embedding shape: {audio_embed.shape}")
    print(f"✓ Audio embedding sample: {audio_embed[0, :5]}")
    
except Exception as e:
    print(f"✗ Error loading audio model: {e}")

# Test ULTRASOUND model
print("\n2. ULTRASOUND MODEL (NTS-Net)")
print("-" * 80)
try:
    us_model = ntsnet_without_head("checkpoints/ultrasound/ultrasound_best.pth", device="cuda")
    print("✓ Ultrasound model loaded successfully")
    
    # Test inference
    test_us = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        us_embed = us_model(test_us)
    print(f"✓ Ultrasound embedding shape: {us_embed.shape}")
    print(f"✓ Ultrasound embedding sample: {us_embed[0, :5]}")
    
except Exception as e:
    print(f"✗ Error loading ultrasound model: {e}")

# Check checkpoint structure
print("\n3. CHECKPOINT STRUCTURE")
print("-" * 80)
try:
    audio_ckpt = torch.load("checkpoints/audio/audio_best.pth", map_location='cpu')
    print(f"✓ Audio checkpoint keys: {list(audio_ckpt.keys())}")
    if "model_state_dict" in audio_ckpt:
        print(f"✓ Model state dict present with {len(audio_ckpt['model_state_dict'])} parameters")
        first_param = list(audio_ckpt['model_state_dict'].items())[0]
        print(f"✓ First parameter: {first_param[0]} with shape {first_param[1].shape}")
except Exception as e:
    print(f"✗ Error reading checkpoint: {e}")

print("\n" + "=" * 80)