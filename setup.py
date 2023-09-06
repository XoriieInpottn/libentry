#!/usr/bin/env python3

from setuptools import setup

if __name__ == '__main__':
    with open('README.md') as file:
        long_description = file.read()
    setup(
        name='libexperiment',
        packages=[
            'libexperiment',
        ],
        version='1.0',
        description='Utilities to perform and re-product experiments.',
        long_description_content_type='text/markdown',
        long_description=long_description,
        license='Apache-2.0 license ',
        author='xi',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/libexperiment',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[]
    )
