#
# Copyright (c) 2020 The rlutils authors
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fr:
    req_pgks_list = fr.readlines()

setuptools.setup(
    name="rlutils",
    version="0.0.1",
    author="The rlutils authors",
    author_email="lucas_lehnert@brown.edu",
    description="rlutils: A utility package for implementing reinforcement learning simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucaslehnert/rlutils",
    packages=setuptools.find_packages(include=['rlutils*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=req_pgks_list,
    python_requires='>=3.6'
)
