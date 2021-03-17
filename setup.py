import pathlib
import setuptools
from setuptools import setup
from setuptools.command.install import install
import pkg_resources

with pathlib.Path('requirements.txt').open() as requirements_txt:
    reqs = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]
    
setuptools.setup(
    name="CKG", # Replace with your own username
    version="1.0.0",
    author="Alberto Santos Delgado",
    author_email="alberto.santos@sund.ku.dk",
    description="A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.",
    url="https://github.com/MannLabs/CKG",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    entry_points='''
        [console_scripts]
        ckg_app=ckg.report_manager.index:main
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.9',
    include_package_data=True,
)
