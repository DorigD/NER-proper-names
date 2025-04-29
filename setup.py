from setuptools import setup, find_packages

setup(
    name="ner_proper_names",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "datasets"
    ]
)