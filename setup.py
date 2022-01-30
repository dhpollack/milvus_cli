from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="milvus_cli",
    version="0.2.0beta2",
    author="Milvus Team",
    author_email="milvus-team@zilliz.com",
    url="https://github.com/milvus-io/milvus_cli",
    description="CLI for Milvus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=[
        "typer==0.4.0",
        "pydantic==1.9.0",
        "pymilvus==2.0.0",
        "rich==11.1.0",
        "requests==2.27.1",
        "tabulate==0.8.9",
    ],
    entry_points={
        "console_scripts": [
            "milvus_cli = milvus_cli.scripts.milvus_cli:runCliPrompt",
            "milvus_cli2 = milvus_cli.scripts.milvus_cli2:runRepl",
        ],
    },
    python_requires=">=3.8",
)
