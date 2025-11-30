import torch
from src.model.training_pipeline import UnifiedCurriculumDice

def test_loss():
    print("Testing UnifiedCurriculumDice...")
    
    # Initialize loss
    loss_fn = UnifiedCurriculumDice(beta=2.0, lambda_tail=1.0)
    
    # Create dummy data
    # Batch size 2, 1 channel, 10x10x10 volume
    pred = torch.rand(2, 1, 10, 10, 10, requires_grad=True)
    target = torch.rand(2, 1, 10, 10, 10)
    
    # Test with different alphas
    alphas = [0.5, 0.2, 0.0]
    
    for alpha in alphas:
        print(f"\nTesting with alpha={alpha}")
        loss = loss_fn(pred, target, alpha)
        print(f"Loss: {loss.item()}")
        
        # Check gradients
        loss.backward()
        print("Gradients computed successfully.")
        
        # Reset gradients
        pred.grad.zero_()

    print("\nTest passed!")

if __name__ == "__main__":
    test_loss()
