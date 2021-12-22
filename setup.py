#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/21 上午11:53  
@Author: Rocsky
@Project: dllib
@File: setup.py
@Version: 0.1
@Description:
"""
import os
import re
from setuptools import setup, find_packages

root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open(os.path.join(root_dir, 'requirements.txt')) as f:
    requirements = f.read().split()

with open(os.path.join(root_dir, 'README.md'), 'r') as f:
    readme = f.read()

setup(
    name='dllib',
    version='0.1.0',
    description='A deep learning framework based on Pytorch',
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Rocsky Lu",
    author_email='pengchenglu@zju.edu.cn',
    maintainer='Rocsky Lu',
    url='https://github.com/linzhenyuyuchen/ifree',
    # packages=find_packages(include=['configs', 'core', 'datasets', 'networks', 'optimizers',
    #                                 'tools', 'utils']),
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords=['deep learning', 'Pytorch'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: '
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)
