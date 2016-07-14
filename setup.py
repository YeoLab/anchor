#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    'pytest'
]

setup(
    name='anchor-bio',
    version='0.1.0',
    description="Anchor is a python package to estimate modality of splicing, percent methylated, any data that is normalized between 0 and 1",
    long_description=readme + '\n\n' + history,
    author="Olga Botvinnik",
    author_email='olga.botvinnik@gmail.com',
    url='https://github.com/YeoLab/anchor',
    packages=[
        'anchor',
    ],
    package_dir={'anchor-bio':
                 'anchor'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='anchor',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
