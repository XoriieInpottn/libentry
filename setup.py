#!/usr/bin/env python3

__author__ = "xi"

from setuptools import setup

if __name__ == '__main__':
    with open('README.md') as file:
        long_description = file.read()
    setup(
        name='libentry',
        packages=[
            'libentry',
            "libentry.service"
        ],
        version='1.17',
        description='Entries for experimental utilities.',
        long_description_content_type='text/markdown',
        long_description=long_description,
        license='Apache-2.0 license ',
        author='xi',
        author_email='gylv@mail.ustc.edu.cn',
        url='https://github.com/XoriieInpottn/libentry',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            "requests",
            "pydantic",
            "json5",
            "PyYAML",
            "Flask",
            "gunicorn",
        ]
    )
