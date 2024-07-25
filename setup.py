#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="tamer",
    version="0.0.1",
    description="",
    author="Jianhua Zhu",
    author_email="zhujianhuapku@pku.edu.cn",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
