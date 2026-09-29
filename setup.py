from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="tomato",
    version="0.4.0",
    description="ToMATo: Topological Mode Analysis Tool",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pynndescent",
        "matplotlib",
    ],
    url="",
)
