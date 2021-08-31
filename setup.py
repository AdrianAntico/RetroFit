# Copyright (C) 2019-2020 Moez Ali <moez.ali@queensu.ca>
# License: MIT, moez.ali@queensu.ca

import pathlib
from setuptools import setup, find_packages
import os

# The directory containing this file
HERE = os.path.dirname(os.path.abspath("__file__"))

with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()
with open(os.path.join(HERE, "requirements.txt")) as f:
    required = f.read().splitlines()

setup(
    name="retrofit",
    version="0.0.6",
    description="AutoML, Forecasting, NLP, Image Classification, Feature Engineering, Model Evaluation, Model Interpretation, Fast Processing.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/AdrianAntico/retrofit",
    author="Adrian Antico",
    author_email="adrianantico@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ]
)
