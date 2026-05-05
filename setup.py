"""Setup script for Real-Time Face Emotion Recognition."""

from setuptools import setup, find_packages

setup(
    name="emotion-recognition",
    version="1.0.0",
    author="Sai Ram Reddy",
    description="Real-Time Face Emotion Recognition using OpenCV and CNN (Keras)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
