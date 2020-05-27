import os

version = "1.0b4"

cwd = os.path.abspath(os.path.dirname(__file__))
graphdb_builder_log = os.path.join(cwd, "graphdb_builder_log.config")
graphdb_connector_log = os.path.join(cwd, "graphdb_connector_log.config")
report_manager_log = os.path.join(cwd, "report_manager_log.config")
connector_config_file = "connector_config.yml"
builder_config_file = "builder/builder_config.yml"
ontologies_config_file = "ontologies/ontologies_config.yml"
databases_config_file = "databases/databases_config.yml"
experiments_config_file = "experiments/experiments_config.yml"
docs_url = "../../docs/_build/html/"
users_config_file = "users/users_config.yml"
