from setuptools import setup, find_packages

setup(
    name="zoomds",
    version="0.60",
    author="Matthew Iversen",
    author_email="iversenmatt7@gmail.com",
    description="A collection of data science and analysis functions for data cleaning, analysis, modeling, and visualization.",
    packages=find_packages(),
    install_requires=[
        "colorlog",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyyaml",
    ],
    python_requires=">=3.6",
)
