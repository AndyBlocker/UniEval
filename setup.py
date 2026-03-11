from setuptools import setup, find_packages

setup(
    name="unieval",
    version="0.1.0",
    description="Universal Evaluation Framework for SNN Conversion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "torchvision",
        "timm==0.3.2",
        "numpy",
        "scipy",
        "pyyaml",
    ],
)
