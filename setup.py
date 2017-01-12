# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 16:56:14 2017

@author: wzswan
"""

try:
    from setuptools import setup

    setup  
    setuptools_kwargs = {
        'install_requires': [
            'numpy',
            'scipy',
            'cvxopt',
        ],
        'provides': ['mysvm'],
    }
except ImportError:
    from distutils.core import setup

    setuptools_kwargs = {}

setup(name='mysvm',
      version="1.0",
      description=(
          """
          Implementations of various
           support vector machine approaches
          """
      ),
      author='Wang zhongsheng',
      author_email='wzswan@gmail.com',
      packages=['mysvm'],
      platforms=['linux'],
      scripts=[],
      **setuptools_kwargs)