"""
Script to check if a PyTorch checkpoint file is valid
"""
import torch
from pathlib import Path
import sys

def check_checkpoint(checkpoint_path: str):
    """Check if a checkpoint file is valid and print its structure"""
    checkpoint_path_obj = Path(checkpoint_path)
    
    print("=" * 80)
    print("PyTorch Checkpoint Validator")
    print("=" * 80)
    print(f"Checkpoint path: {checkpoint_path}")
    print()
    
    # Check if file exists
    if not checkpoint_path_obj.exists():
        print("❌ ERROR: Checkpoint file does not exist!")
        return False
    
    # Check file size
    file_size = checkpoint_path_obj.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    print(f"✓ File exists")
    print(f"  File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    
    if file_size == 0:
        print("❌ ERROR: Checkpoint file is empty!")
        return False
    
    if file_size < 1024:  # Less than 1KB
        print("⚠️  WARNING: Checkpoint file is suspiciously small (< 1KB)")
        print("   This might indicate the file is corrupted or incomplete")
    
    # Try to read first few bytes to check if it's a binary file
    print()
    print("Checking file format...")
    try:
        with open(checkpoint_path, 'rb') as f:
            first_bytes = f.read(10)
            print(f"  First 10 bytes (hex): {first_bytes.hex()}")
            print(f"  First 10 bytes (repr): {repr(first_bytes)}")
            
            # PyTorch files typically start with specific magic bytes
            # Older format: starts with pickle protocol bytes
            # Newer format: starts with ZIP file signature (PK\x03\x04)
            if first_bytes.startswith(b'PK'):
                print("  ✓ Appears to be a ZIP-based PyTorch checkpoint (newer format)")
            elif first_bytes.startswith(b'\x80'):
                print("  ✓ Appears to be a pickle-based PyTorch checkpoint (older format)")
            else:
                print("  ⚠️  WARNING: File doesn't start with expected PyTorch magic bytes")
                print("     This might not be a valid PyTorch checkpoint")
    except Exception as e:
        print(f"  ❌ ERROR reading file: {e}")
        return False
    
    # Try to load the checkpoint
    print()
    print("Attempting to load checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("  ✓ Successfully loaded checkpoint!")
    except Exception as e:
        print(f"  ❌ ERROR loading checkpoint: {e}")
        print(f"     Error type: {type(e).__name__}")
        return False
    
    # Check checkpoint structure
    print()
    print("Checkpoint structure:")
    print("-" * 80)
    
    if isinstance(checkpoint, dict):
        print(f"  Type: Dictionary with {len(checkpoint)} keys")
        print()
        print("  Keys found:")
        for key in sorted(checkpoint.keys()):
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: Tensor {list(value.shape)} (dtype: {value.dtype})")
            elif isinstance(value, dict):
                if 'state_dict' in str(key).lower() or 'state' in str(key).lower():
                    print(f"    - {key}: Dictionary with {len(value)} parameters")
                    # Show first few keys
                    sample_keys = list(value.keys())[:5]
                    print(f"      Sample keys: {sample_keys}")
                    if len(value) > 5:
                        print(f"      ... and {len(value) - 5} more")
                else:
                    print(f"    - {key}: Dictionary with {len(value)} items")
            elif isinstance(value, (int, float, str)):
                print(f"    - {key}: {type(value).__name__} = {value}")
            else:
                print(f"    - {key}: {type(value).__name__}")
        
        # Check for common checkpoint keys
        print()
        print("  Required keys check:")
        required_keys = ['student_state_dict', 'config']
        optional_keys = ['teacher_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'global_step']
        
        for key in required_keys:
            if key in checkpoint:
                print(f"    ✓ {key}: Found")
            else:
                print(f"    ❌ {key}: Missing (required for evaluation)")
        
        for key in optional_keys:
            if key in checkpoint:
                print(f"    ✓ {key}: Found")
            else:
                print(f"    - {key}: Not found (optional)")
        
        # Check student_state_dict structure if it exists
        if 'student_state_dict' in checkpoint:
            print()
            print("  student_state_dict analysis:")
            state_dict = checkpoint['student_state_dict']
            print(f"    Total parameters: {len(state_dict)}")
            
            # Count parameters by type
            backbone_keys = [k for k in state_dict.keys() if 'backbone' in k or not any(x in k for x in ['head', 'projector', 'mlp'])]
            head_keys = [k for k in state_dict.keys() if 'head' in k]
            projector_keys = [k for k in state_dict.keys() if 'projector' in k or 'mlp' in k]
            
            print(f"    Backbone parameters: {len(backbone_keys)}")
            if head_keys:
                print(f"    Head parameters: {len(head_keys)}")
            if projector_keys:
                print(f"    Projector/MLP parameters: {len(projector_keys)}")
            
            # Show sample backbone keys
            if backbone_keys:
                print()
                print("    Sample backbone parameter keys:")
                for key in backbone_keys[:10]:
                    shape = list(state_dict[key].shape) if isinstance(state_dict[key], torch.Tensor) else "N/A"
                    print(f"      - {key}: {shape}")
                if len(backbone_keys) > 10:
                    print(f"      ... and {len(backbone_keys) - 10} more")
        
        # Check config if it exists
        if 'config' in checkpoint:
            print()
            print("  config analysis:")
            config = checkpoint['config']
            if isinstance(config, dict):
                print(f"    Config type: Dictionary with {len(config)} top-level keys")
                if 'model' in config:
                    print(f"    ✓ Model config found")
                if 'training' in config:
                    print(f"    ✓ Training config found")
            else:
                print(f"    Config type: {type(config).__name__}")
    
    else:
        print(f"  Type: {type(checkpoint).__name__}")
        if isinstance(checkpoint, torch.nn.Module):
            print("  This appears to be a model state dict (not a full checkpoint)")
        else:
            print("  ⚠️  WARNING: Checkpoint is not a dictionary")
    
    print()
    print("=" * 80)
    print("✓ Checkpoint validation complete!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <path_to_checkpoint.pth>")
        print()
        print("Example:")
        print("  python check_checkpoint.py checkpoints/checkpoint_epoch_160.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = check_checkpoint(checkpoint_path)
    
    if success:
        print("\n✓ Checkpoint appears to be valid!")
        sys.exit(0)
    else:
        print("\n❌ Checkpoint validation failed!")
        sys.exit(1)

