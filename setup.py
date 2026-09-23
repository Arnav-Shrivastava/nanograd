from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nanograd",
    version="0.1.0",
    author="Arnav Shrivastava",
    description="A tiny scalar-valued autograd engine built from scratch — a learning project inspired by Karpathy's micrograd.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arnav-Shrivastava/nanograd",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[],   # zero runtime dependencies — pure Python!
    extras_require={
        "dev": [
            "pytest>=7.0",
            "graphviz",        # for computational graph visualisation
            "matplotlib",      # for the demo notebooks
            "numpy",
        ]
    },
)
