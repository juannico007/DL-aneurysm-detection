import torch
from src.model.training_pipeline import UnifiedCurriculumDice

def test_max_loss():
    print("Testing Background Max Suppression...")
    
    # Setup
    # Batch size 1, 1 channel, 10x10x10 volume
    pred = torch.zeros((1, 1, 10, 10, 10), dtype=torch.float32)
    target = torch.zeros((1, 1, 10, 10, 10), dtype=torch.float32)
    
    # Alpha = 0.5
    alpha = 0.5
    
    # Case 1: Perfect prediction (all zeros)
    criterion = UnifiedCurriculumDice(lambda_max=1.0)
    loss = criterion(pred, target, alpha)
    print(f"Loss (Perfect): {loss.item()}")
    # Dice should be 1.0 (perfect intersection of 0? No, intersection of 0 is tricky with epsilon)
    # Wait, intersection of zeros with zeros...
    # target is all zeros. core_mask = (target >= alpha) -> all zeros.
    # tail_mask = 1.0.
    # p_core = 0, t_core = 0. Intersection = 0. Diff = 0.
    # tail_penalty = relu(0 - 0.5) = 0.
    # Denom = epsilon. Dice = epsilon / epsilon = 1.0. Loss = 0.0.
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
    
    # Case 2: Single high-value background voxel
    # Set one voxel to 0.9 (above alpha)
    pred[0, 0, 5, 5, 5] = 0.9
    
    # Calculate expected loss
    # Dice part:
    # p_core = 0 (since core_mask is 0). Intersection = 0.
    # tail_penalty = (0.9 - 0.5)^2 = 0.4^2 = 0.16.
    # Weighted tail error = 0.16.
    # Denom = 0 + 0 + 1.0 * 0.16 + epsilon = 0.16 + eps.
    # Dice = eps / (0.16 + eps) approx 0.
    # Dice Loss approx 1.0.
    
    # Max Loss part:
    # tail_probs = pred * tail_mask = pred (since tail_mask is 1).
    # max_tail_error = 0.9.
    # Total Loss = DiceLoss + lambda_max * 0.9.
    
    loss_1 = criterion(pred, target, alpha)
    print(f"Loss (1 voxel 0.9, lambda=1.0): {loss_1.item()}")
    
    # Case 3: Increase lambda_max
    criterion_2 = UnifiedCurriculumDice(lambda_max=10.0)
    loss_2 = criterion_2(pred, target, alpha)
    print(f"Loss (1 voxel 0.9, lambda=10.0): {loss_2.item()}")
    
    # Difference should be roughly 9 * 0.9 = 8.1
    diff = loss_2 - loss_1
    print(f"Difference: {diff.item()}")
    assert torch.isclose(diff, torch.tensor(8.1), atol=1e-2)
    
    print("Background Max Suppression Test Passed!")

if __name__ == "__main__":
    test_max_loss()
