from setuptools import setup, find_packages

setup(
    name="graphistry-mcp-server",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest",
        "pytest-asyncio",
    ],
    python_requires=">=3.10",
) 