import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
import ckg.init


with open("README.rst", "r") as fh:
    long_description = fh.read()

class PreInstallCommand(install):
    """Pre-installation for install mode."""
    def run(self):
        check_call("pip install -r requirements.txt".split())
        ckg.init.installer_script()
        install.run(self)
        
class PreDevelopCommand(develop):
    """Pre-installation for install mode."""
    def run(self):
        check_call("pip install -r requirements.txt".split())
        ckg.init.installer_script()
        develop.run(self)


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
    cmdclass={
        'develop': PreDevelopCommand,
        'install': PreInstallCommand,
    },
    entry_points={'console_scripts': [
        'ckg_app=ckg.report_manager.index:main',
        'ckg_debug=ckg.debug:main',
        'ckg_build=ckg.graphdb_builder.builder.builder:run_full_update',
        'ckg_update_textmining=ckg.graphdb_builder.builder.builder:update_textmining']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.9',
    include_package_data=True,
)
