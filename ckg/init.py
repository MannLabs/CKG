"""
init file

"""
import os

version = "1.0"

cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.abspath(os.path.join(cwd, os.pardir))


def set_paths(config_path):
    """
    Creates path for CKG if they do not exists
    """
    structure_directory = '.'
    data_directory_structure = {"data": [
                                    "archive",
                                    "databases",
                                    "experiments",
                                    "ontologies",
                                    "users",
                                    "stats",
                                    "downloads",
                                    "reports",
                                    "tmp",
                                    "imports",
                                    "imports/databases",
                                    "imports/experiments",
                                    "imports/ontologies",
                                    "imports/users",
                                    "imports/curated"
                                    ],
                                "log": []
                                }    

    for directory in data_directory_structure:
        new_dir = os.path.join(cwd, directory)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        with open(os.path.join(config_path, 'ckg_config.yml'), 'a') as c:
            c.write(directory+'_directory: "'+new_dir.replace('\\', '/')+'"\n')
            for int_dir in data_directory_structure[directory]:
                new_int_dir = os.path.join(new_dir, int_dir)
                if not os.path.exists(new_int_dir):
                    os.makedirs(new_int_dir)
        
                c.write(int_dir.replace('/', '_')+'_directory: "'+new_int_dir.replace('\\', '/')+'"\n')

def setup_config_file(file_name, path):
    with open(file_name, 'r') as f:
        config = f.read()
        config = config.replace('{PATH}', path).replace('\\', '/')
    with open(file_name.replace('_template', ''), 'w') as out:
        out.write(config)

    return file_name.replace('_template', '')


def setup_config_files(config_path):
    config_files = {'report_manager_log': 'report_manager_log_template.config', 
                    'graphdb_connector_log': 'graphdb_connector_log_template.config',
                    'graphdb_builder_log':'graphdb_builder_log_template.config',
                    'analytics_factory_log':'analytics_factory_log_template.config'}
    for log_file in config_files:
        config_file = config_files[log_file]
        file_name = os.path.join(config_path, config_file)
        nfile = setup_config_file(file_name, cwd)

        with open(os.path.join(config_path, 'ckg_config.yml'), 'a') as c:
            c.write(log_file+': "'+ nfile.replace('\\', '/') + '"\n')
        
        
def installer_script():
    config_path = os.path.join(cwd, 'ckg/config/')
    with open(os.path.join(config_path, 'ckg_config.yml'), 'w') as c:
        c.write('version: '+version+'\n')
        c.write('ckg_directory: "'+os.path.join(cwd, 'ckg').replace('\\', '/')+'"\n')

    set_paths(config_path)
    setup_config_files(config_path)
    
    print("DONE DOING THE STUFF")


if __name__ == "__main__":
    installer_script()
    print('CKG config paths set.')
