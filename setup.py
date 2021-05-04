import os
import pathlib
import setuptools
from setuptools import setup
from setuptools.command.install import install
import pkg_resources
import ckg.init


with pathlib.Path('requirements.txt').open() as requirements_txt:
    reqs = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

ckg.init.installer_script()

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="CKG", # Replace with your own username
    version="1.0.0",
    author="Alberto Santos Delgado",
    author_email="alberto.santos@sund.ku.dk",
    description="A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url="https://github.com/MannLabs/CKG",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    entry_points={'console_scripts': [
        'ckg_app=ckg.report_manager.index:main',
        'ckg_debug=ckg.debug:main']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.9',
    include_package_data=True,
)
