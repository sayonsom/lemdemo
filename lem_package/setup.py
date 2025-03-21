from setuptools import setup, find_packages

setup(
    name="lem_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pydantic>=1.10.7",
    ],
    python_requires=">=3.8",
    description="Large Event Model (LEM) for processing and predicting timeseries events from home appliances",
    author="LEM Team",
    include_package_data=True,
    package_data={
        "lem_model": ["*.pt", "*.pkl", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "lem-cli=lem_package.cli:main",
            "lem-api=lem_package.api:run_server",
        ],
    },
) 