'''
Copyright (C) 2018-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

from pathlib import Path

from setuptools import find_packages, setup

VERSION = '0.7.1.2'

requirements = [
    'torch',
]

SETUP_DIR = Path(__file__).resolve().parent

TEST_BASE_EXTRAS = (SETUP_DIR / 'test_requirements.txt').read_text()
EXTRAS_REQUIRE = {
    'dev': TEST_BASE_EXTRAS,
}

setup(
    name='ptflops',
    version=VERSION,
    author='Vladislav Sovrasov',
    author_email='sovrasov.vlad@gmail.com',
    url='https://github.com/sovrasov/flops-counter.pytorch',
    description='Flops counter for convolutional networks in'
                'pytorch framework',
    long_description=(SETUP_DIR / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    license='MIT',

    packages=find_packages(SETUP_DIR, exclude=('*test*',)),
    package_dir={'ptflops': str(SETUP_DIR / 'ptflops')},

    zip_safe=True,
    install_requires=requirements,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.7',

    classifiers=[
        'MIT Software License :: Programming Language :: Python :: 3',
    ],
)
