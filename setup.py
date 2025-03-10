"""
Setup script for the Sentence Transformer Multi-Task Learning project.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="sentence-multitask",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A multi-task learning system using sentence transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FetchProject",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-model=src.run_model:main",
            "run-model-real=src.run_model_real_data:main",
            "run-api=src.run_api:main",
            "run-frontend=src.run_frontend:main",
        ],
    },
) 