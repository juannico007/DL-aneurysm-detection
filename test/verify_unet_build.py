
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.unet import UNet

def test_unet_build():
    print("Testing UNet with up_mode='trilinear'...")
    try:
        model = UNet(
            in_channels=1,
            out_channels=1,
            n_blocks=2,
            start_filters=4,
            up_mode='trilinear'
        )
        x = torch.randn(1, 1, 32, 32, 32)
        y, _ = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        
        if y.shape == x.shape:
            print("SUCCESS: Output shape matches input shape.")
        else:
            print("FAILURE: Output shape mismatch!")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_unet_build()
