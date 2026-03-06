from setuptools import setup, find_packages

setup(
    name="egtb_moe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
    ],
    author="Anonymous",
    description="Entropy-Gated Thinking Budgets for Sparse MoE",
    python_requires=">=3.8",
)
