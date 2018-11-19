import yaml

def read_yaml(yaml_file):
    content = None
    with open(yaml_file, 'r') as stream:
        try:
            content = yaml.load(stream)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
    return content

def get_configuration(configuration_file):
    configuration  = None
    if configuration_file.endswith("yml"):
        configuration = read_yaml(configuration_file)
    else:
        raise Exception("The format specified in the configuration file {} is not supported. {}".format(configuration_file, err))

    return configuration

def get_configuration_variable(configuration_file, variable):
    configuration = get_configuration(configuration_file)
    if variable in configuration:
        return configuration[variable]
    else:
        raise Exception("The varible {} is not found in the configuration file {}. {}".format(variable, configuration_file, err))
