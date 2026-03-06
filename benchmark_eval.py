import torch
from egtb_core import EGTBMoELayer
import time

def run_benchmark():
    print("Initializing EGTB Benchmark (196B MoE Foundation Simulation)...")
    n_batch, n_seq, n_embed = 8, 512, 4096
    layer = EGTBMoELayer(n_embed, n_experts=128, k_max=8)
    
    dummy_input = torch.randn(n_batch, n_seq, n_embed)
    
    start_time = time.time()
    with torch.no_grad():
        _ = layer(dummy_input)
    end_time = time.time()
    
    print(f"Inference complete in {end_time - start_time:.4f}s")
    print(f"Final Adjusted Threshold (tau): {layer.controller.tau:.4f}")
    print("Target Sparsity: 0.5 | Logic: Entropy-Gated Adaptive Routing [cite: 7]")

if __name__ == "__main__":
    run_benchmark()
