from setuptools import setup, find_packages
import os

# Get the directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Read README.md with error handling
try:
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Federated Machine Learning for Privacy-Preserving Smart Healthcare Applications"

# Read requirements.txt with error handling
try:
    with open(os.path.join(here, "requirements.txt"), "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    name="federated-ml-healthcare",
    version="0.1.0",
    author="Federated ML Healthcare Contributors",
    author_email="your-email@example.com",
    description="Federated Machine Learning for Privacy-Preserving Smart Healthcare Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reethu-s123/federated-ml-healthcare",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
        ],
    },
)
