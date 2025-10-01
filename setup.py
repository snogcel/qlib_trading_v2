"""
Setup script for Trading Pipeline CLI
"""

from setuptools import setup, find_packages

setup(
    name="trading-pipeline-cli",
    version="1.0.0",
    description="CLI application for quantile trading model training and backtesting",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.2.0",
        "optuna>=2.10.0",
        "qlib",
        "stable-baselines3",
    ],
    entry_points={
        "console_scripts": [
            "trading-cli=src.cli.trading_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)