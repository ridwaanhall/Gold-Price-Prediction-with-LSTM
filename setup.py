"""
Setup configuration for the Gold Price Prediction package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
__version__ = "1.0.0"

setup(
    name="gold-price-prediction",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready LSTM-based gold price prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gold-price-prediction",
    packages=find_packages(exclude=["tests*", "notebooks*", "deployment*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "api": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "pydantic>=1.10.0",
        ],
        "monitoring": [
            "wandb>=0.13.0",
            "mlflow>=1.30.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.4.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "pydantic>=1.10.0",
            "wandb>=0.13.0",
            "mlflow>=1.30.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gold-predict=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml"],
        "data": ["*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gold-price-prediction/issues",
        "Source": "https://github.com/yourusername/gold-price-prediction",
        "Documentation": "https://gold-price-prediction.readthedocs.io/",
        "Changelog": "https://github.com/yourusername/gold-price-prediction/blob/main/CHANGELOG.md",
    },
    keywords=[
        "machine learning",
        "lstm",
        "time series",
        "forecasting",
        "gold price",
        "financial prediction",
        "deep learning",
        "tensorflow",
        "neural networks",
        "cryptocurrency",
        "trading",
        "investment",
    ],
    zip_safe=False,
)
