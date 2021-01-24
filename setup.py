import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import os

def set_paths():
    """
    Creates path for CKG if they do not exists
    """
    path = os.path.dirname(os.path.abspath(__file__))
    structure_directory = '.'
    data_directory_structure = {"data": [
                                    "archive",
                                    "databases",
                                    "experiments",
                                    "imports/databases",
                                    "imports/experiments",
                                    "imports/ontologies",
                                    "imports/stats",
                                    "ontologies",
                                    "tmp"
                                    ],
                                "log": []
                                }

    for directory in data_directory_structure:
        new_dir = os.path.join(os.path.join(path, structure_directory), directory)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for int_dir in data_directory_structure[directory]:
            new_int_dir = os.path.join(new_dir, int_dir)
            if not os.path.exists(new_int_dir):
                os.makedirs(new_int_dir)

def setup_config_file(file_name, path):
    with open(file_name, 'r') as f:
        config = f.read()
        config = config.replace('{PATH}', path).replace('\\', '/')
    with open(file_name.replace('_template', ''), 'w') as out:
        out.write(config)

def setup_config_files():
    config_dir = 'ckg/config/'
    config_files = ['report_manager_log_template.config', 'graphdb_connector_log_template.config', 'graphdb_builder_log_template.config']
    path = os.path.dirname(os.path.abspath(__file__))
    for config_file in config_files:
        file_name = os.path.join(path, os.path.join(config_dir, config_file))
        setup_config_file(file_name, path)


def installer_script():
    set_paths()
    setup_config_files()
    
#PostDevelopCommand taken from https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        installer_script()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        installer_script()

setuptools.setup(
    name="CKG", # Replace with your own username
    version="1.0.0",
    author="Alberto Santos Delgado",
    author_email="alberto.santos@sund.ku.dk",
    description="A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.",
    url="https://github.com/MannLabs/CKG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
