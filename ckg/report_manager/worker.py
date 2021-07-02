import pandas as pd
from celery import Celery
from ckg.report_manager.apps import projectCreation, dataUpload
from ckg.graphdb_builder.builder import builder
from ckg.graphdb_connector import connector
from ckg.report_manager import project


celery_app = Celery('create_new_project')

celery_app.conf.update(broker_url='redis://127.0.0.1:6379', result_backend='redis://127.0.0.1:6379/0')

@celery_app.task
def create_new_project(identifier, data, separator='|'):
    driver = connector.getGraphDatabaseConnectionConfiguration()
    project_result, projectId = projectCreation.create_new_project(driver, identifier, pd.read_json(data), separator=separator)
    if projectId is not None:
        result = {str(projectId): str(project_result)}
    else:
        result = {}

    return result


@celery_app.task
def create_new_identifiers(project_id, data, directory, filename):
    driver = connector.getGraphDatabaseConnectionConfiguration()
    upload_result = dataUpload.create_experiment_internal_identifiers(driver, project_id, pd.read_json(data, dtype={'subject external_id': object, 'biological_sample external_id': object, 'analytical_sample external_id': object}), directory, filename)
    res_n = dataUpload.check_samples_in_project(driver, project_id)

    return {str(project_id): str(upload_result), 'res_n': res_n.to_dict()}


@celery_app.task
def generate_project_report(project_id, config_files, force):
    p = project.Project(project_id, datasets={}, knowledge=None, report={}, configuration_files=config_files)
    p.build_project(force)
    p.generate_report()

    return {str(p.identifier): "Done"}


@celery_app.task
def run_minimal_update_task(username):
    response = builder.run_minimal_update(user=username, n_jobs=1)

    return {'response': str(response)}

@celery_app.task
def run_full_update_task(username, download):
    response = builder.run_full_update(user=username, download=download)

    return {'response': str(response)}
