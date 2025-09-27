# setup.py
from setuptools import setup, find_packages

setup(
    name="oil-price-prediction-wavelets",
    version="1.0.0",
    description="Oil Price Prediction using Wavelets and TensorFlow 2.0",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AI Assistant",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyWavelets>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "yfinance>=0.2.0",
        "scipy>=1.7.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)