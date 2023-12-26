'''
Copyright (C) 2018-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

from pathlib import Path

from setuptools import find_packages, setup

VERSION = '0.7.2.1'

requirements = [
    'torch',
]

SETUP_DIR = Path(__file__).resolve().parent
TEST_REQ = SETUP_DIR / 'test_requirements.txt'

if TEST_REQ.is_file():
    TEST_BASE_EXTRAS = TEST_REQ.read_text()
    EXTRAS_REQUIRE = {
        'dev': TEST_BASE_EXTRAS,
    }
else:
    EXTRAS_REQUIRE = {}

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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
