import torch
import torch.nn as nn
from src.model.training_pipeline import UnifiedCurriculumDice

def test_combined_loss():
    print("Testing Combined Loss (Dice + BCE)...")
    
    # Initialize losses
    dice_loss_fn = UnifiedCurriculumDice(beta=2.0, lambda_tail=1.0)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    seg_weight = 1.0
    cls_weight = 0.5
    
    # Create dummy data
    # Batch size 2, 1 channel, 10x10x10 volume
    pred_seg = torch.rand(2, 1, 10, 10, 10, requires_grad=True)
    target_seg = torch.rand(2, 1, 10, 10, 10)
    
    pred_cls = torch.randn(2, 1, requires_grad=True) # Logits
    target_cls = torch.randint(0, 2, (2, 1)).float()
    
    alpha = 0.5
    
    # Calculate losses
    dice_loss = dice_loss_fn(pred_seg, target_seg, alpha)
    bce_loss = bce_loss_fn(pred_cls, target_cls)
    
    total_loss = seg_weight * dice_loss + cls_weight * bce_loss
    
    print(f"Dice Loss: {dice_loss.item()}")
    print(f"BCE Loss: {bce_loss.item()}")
    print(f"Total Loss: {total_loss.item()}")
    
    # Check gradients
    total_loss.backward()
    print("Gradients computed successfully.")
    
    print("\nTest passed!")

if __name__ == "__main__":
    test_combined_loss()
