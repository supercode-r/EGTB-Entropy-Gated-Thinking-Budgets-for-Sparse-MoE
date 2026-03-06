# EGTB: Entropy-Gated Thinking Budgets for Sparse MoE

[![Paper Status](https://img.shields.io/badge/Status-EAI%20CSECS%202026-blue.svg)](./paper/main.pdf)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](./egtb_core.py)

Official repository for **"Adaptive Computational Scaling in Sparse Mixture-of-Experts via Entropy-Gated Thinking Budgets"**.

## 🧠 Overview
EGTB addresses the **Uniform Sparsity Paradox** [cite: 15] by dynamically modulating activated parameters based on real-time entropy metrics[cite: 7].

### Key Equations
* **Entropy Sensing ($H$):** $H(x) = -\sum_{i=1}^{N} p_i \log_2 p_i$ [cite: 42]
* **Varentropy ($V$):** $V(x) = \sum_{i=1}^{N} p_i (\log_2 p_i + H(x))^2$ [cite: 49]
* **PI-Controller ($\tau$):** $\tau_{t+1} = \tau_t + K_p e_t + K_i \int e_t dt$ [cite: 56]

## 📊 Performance
* **Efficiency:** 32% Average Active FLOP reduction.
* **Accuracy:** +4.2% on LiveCodeBench (Zero-shot).
* **Reasoning:** +15% completion on DeepPlanning long-horizon tasks[cite: 98].

## 🛠 Usage
```python
from egtb_core import EGTBMoELayer
model = EGTBMoELayer(n_embed=4096, n_experts=128, target_sparsity=0.7)
output = model(hidden_states)
