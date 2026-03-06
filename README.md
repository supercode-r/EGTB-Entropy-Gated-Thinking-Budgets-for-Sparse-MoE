# EGTB-Entropy-Gated-Thinking-Budgets-for-Sparse-MoE
This repository contains the official implementation and research artifacts for "Adaptive Computational Scaling in Sparse Mixture-of-Experts via Entropy-Gated Thinking Budgets."

Overview

EGTB is a novel framework designed to resolve the "Uniform Sparsity Paradox" in Mixture-of-Experts (MoE) models. While traditional MoE models use a fixed $k$ (e.g., Top-2), EGTB uses information-theoretic signals to dynamically allocate compute:

Entropy Gate: Uses Shannon Entropy and Varentropy to sense token difficulty.

PI-Controller: Regulates routing thresholds to maintain global FLOP consistency.

Null-Experts: Implements an "express lane" for functional tokens (punctuation, fillers), bypassing FFN computation.

Repository Structure

egtb/: Core Python implementation (Router, PI-Controller, MoE Layers).

paper/: LaTeX source for the EAI CSECS 2026 submission.

benchmarks/: Scripts for evaluating on DeepPlanning and LiveCodeBench.

Key Findings

32% Reduction in average active FLOPs.

4.2% Improvement in zero-shot coding accuracy.

Successfully manages "Reasoning Exhaustion" in long-trajectory agentic tasks.

Citation

If you use this work in your research, please cite the paper provided in the /paper directory.
