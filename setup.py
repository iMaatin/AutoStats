from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="AutoStats",  
    version="0.1.0",  
    author="iMaatin",
    author_email="2012irajpour@gmail.com",
    description="A Python package for automatic statistical analysis and imputation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/iMaatin/AutoStats", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
