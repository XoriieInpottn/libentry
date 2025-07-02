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
            "libentry.mcp",
            "libentry.service"
        ],
        entry_points={
            'console_scripts': [
                'libentry_test_api = libentry.test_api:main'
            ]
        },
        version='1.24.5',
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
            "pydantic",
            "PyYAML",
            "numpy",
            "urllib3",
            "httpx",
            "Flask",
            "gunicorn",
        ]
    )
