import os

def setup_config_file(file_name, path):
    with open(file_name, 'r') as f:
        config = f.read()
        config = config.replace('{PATH}', path)
    with open(file_name.replace('_template', ''), 'w') as out:
        out.write(config)


if __name__ == '__main__':
    config_dir = 'src/config/'
    config_files = ['report_manager_log_template.config', 'graphdb_connector_log_template.config', 'graphdb_builder_log_template.config']
    path = os.path.dirname(os.path.abspath(__file__))
    for config_file in config_files:
        file_name = os.path.join(path, os.path.join(config_dir, config_file))
        setup_config_file(file_name, path)
        


