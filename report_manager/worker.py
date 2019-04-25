import os
import pandas as pd
import datetime
from celery import Celery
from report_manager.apps import projectCreation
from graphdb_connector import connector


celery_app = Celery('create_new_project')

celery_app.conf.update(broker_url = 'redis://localhost:6379',
					   result_backend = 'redis://localhost:6379/0')


@celery_app.task
def create_new_project(identifier, data, subjects, timepoints):
    driver = connector.getGraphDatabaseConnectionConfiguration()
    project_result, projectId = projectCreation.create_new_project(driver, identifier, pd.read_json(data))
    subject_result = projectCreation.create_new_subject(driver, projectId, subjects)
    timepoint_result = projectCreation.create_new_timepoint(driver, projectId, timepoints)

    return {str(projectId): str(project_result), 'subjects':str(subject_result), 'timepoints':str(timepoint_result)}
