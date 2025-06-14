from setuptools import setup, find_packages

setup(
    name="multi-agent-research-cli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "requests>=2.28.0",
        "colorama>=0.4.6",
    ],
    entry_points={
        'console_scripts': [
            'mas=cli:cli',
        ],
    },
    author="Ricardo Ledan",
    author_email="ricardoledan@proton.me",
    description="Multi-Agent Research System CLI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ricoledan/supervisor-multi-agent-system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)