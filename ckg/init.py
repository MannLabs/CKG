"""
init file

"""
import os
def set_paths():
    """
    Creates path for CKG if they do not exists
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(path, os.pardir))
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
    path = os.path.abspath(os.path.join(path, os.pardir))
    for config_file in config_files:
        file_name = os.path.join(path, os.path.join(config_dir, config_file))
        setup_config_file(file_name, path)


def installer_script():
    set_paths()
    setup_config_files()


if __name__ == "__main__":
    installer_script()
    print('CKG config paths set.')
