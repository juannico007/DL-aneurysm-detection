
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.unet import UNet
from src.model.training_pipeline import custom_collate_with_coords

def test_coord_regression():
    print("Testing UNet with regression head...")
    try:
        # Test with actual input shape (64x64x64)
        batch_size = 2
        input_shape = (batch_size, 1, 64, 64, 64)
        
        model = UNet(
            in_channels=1,
            out_channels=1,
            n_blocks=2,
            start_filters=4,
            up_mode='trilinear',
            regression=True
        )
        x = torch.randn(input_shape)
        y, cls_out, reg_out = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Seg output shape: {y.shape}")
        print(f"Class output shape: {cls_out.shape}")
        print(f"Reg output shape: {reg_out.shape}")
        
        if reg_out.shape == (batch_size, 3):
            print("SUCCESS: Regression output shape is correct (B, 3).")
        else:
            print(f"FAILURE: Regression output shape mismatch! Expected ({batch_size}, 3), got {reg_out.shape}")
            
        # Test collate
        print("\nTesting collate function...")
        batch = [
            (torch.randn(1, 32, 32, 32), torch.randn(1, 32, 32, 32), 1, torch.tensor([0.5, 0.5, 0.5])),
            (torch.randn(1, 32, 32, 32), torch.randn(1, 32, 32, 32), 0, torch.tensor([-1.0, -1.0, -1.0]))
        ]
        X, M, Y, C = custom_collate_with_coords(batch)
        print(f"Collate Coords shape: {C.shape}")
        if C.shape == (2, 3):
             print("SUCCESS: Collate function works.")
        else:
             print("FAILURE: Collate function mismatch.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coord_regression()
