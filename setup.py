import os
import shutil
import sys
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.4'

requirements = [
    'torch',
]

setup(
    # Metadata
    name='ptflops',
    version=VERSION,
    author='Vladislav Sovrasov',
    author_email='sovrasov.vlad@gmail.com',
    url='https://github.com/sovrasov/flops-counter.pytorch',
    description='Flops counter for convolutional networks in pytorch framework',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
