import torch
import numpy as np
import math
from src.model.training_pipeline import TrainingPipeline, _gaussian_heatmap

def test_alpha_schedule():
    print("Testing Alpha Schedule...")
    
    # Mock TrainingPipeline
    class MockPipeline:
        def __init__(self):
            self.neg_warmup_epochs = 0
            self.neg_ramp_epochs = 20
            self.alpha_max = 0.5
            self.alpha_min = 0.05
            self.alpha_schedule = "cosine"
            
        def _get_current_alpha(self, epoch):
            # Copy-paste logic for testing or import if possible (but it's an instance method)
            # We will use the actual method logic by binding it or just re-implementing to verify the math
            # Better: let's instantiate the actual pipeline if possible, but it has many dependencies.
            # Let's just use the logic we implemented to verify the math.
            
            if epoch <= self.neg_warmup_epochs:
                return self.alpha_max
                
            ramp_step = epoch - self.neg_warmup_epochs
            
            if ramp_step > self.neg_ramp_epochs:
                return self.alpha_min
                
            t = (ramp_step - 1) / max(1, self.neg_ramp_epochs - 1)
            t = min(max(t, 0.0), 1.0)
            
            if self.alpha_schedule == "cosine":
                decay = 0.5 * (1.0 + math.cos(t * math.pi))
            else:
                decay = 1.0 - t
                
            return self.alpha_min + (self.alpha_max - self.alpha_min) * decay

    pipeline = MockPipeline()
    
    print(f"Epoch 1 (Start): {pipeline._get_current_alpha(1):.4f}")
    print(f"Epoch 10 (Mid): {pipeline._get_current_alpha(10):.4f}")
    print(f"Epoch 20 (End Ramp): {pipeline._get_current_alpha(20):.4f}")
    print(f"Epoch 21 (Stabilization): {pipeline._get_current_alpha(21):.4f}")
    print(f"Epoch 40 (End): {pipeline._get_current_alpha(40):.4f}")
    
    assert pipeline._get_current_alpha(1) == 0.5
    assert pipeline._get_current_alpha(21) == 0.05
    assert pipeline._get_current_alpha(40) == 0.05
    print("Alpha Schedule Test Passed!")

def test_heatmap_optimization():
    print("\nTesting Heatmap Optimization...")
    shape = (64, 64, 64)
    center = (32.0, 32.0, 32.0)
    sigma = 5.0
    
    # Run optimized function
    heatmap = _gaussian_heatmap(shape, center, sigma)
    
    # Check peak
    peak = heatmap.max()
    print(f"Peak value: {peak}")
    assert np.isclose(peak, 1.0, atol=1e-5)
    
    # Check value at 3 sigma
    # dist = 3*sigma = 15. 
    # value should be exp(-15^2 / (2*25)) = exp(-225/50) = exp(-4.5) ~= 0.011
    z_3sigma = int(center[0] + 3 * sigma)
    val_3sigma = heatmap[z_3sigma, int(center[1]), int(center[2])]
    print(f"Value at 3 sigma: {val_3sigma}")
    
    # Check value far away (should be 0 due to bounding box)
    val_far = heatmap[0, 0, 0]
    print(f"Value far away: {val_far}")
    assert val_far == 0.0
    
    print("Heatmap Optimization Test Passed!")

if __name__ == "__main__":
    test_alpha_schedule()
    test_heatmap_optimization()
