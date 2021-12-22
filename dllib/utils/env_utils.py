#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Time: 2021/12/20 下午4:19  
@Author: Rocsky
@Project: dllib
@File: env_utils.py
@Version: 0.1
@Description:
"""
import os


def get_git_branch() -> str:
    """
    get the branch name of the current git
    :return: git branch name
    """
    branches = os.popen('git branch').readlines()
    for branch in branches:
        if branch.startswith('*'):
            cur_branch = branch.strip().split()[-1]
            break
    return cur_branch
