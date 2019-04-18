import os
import pandas as pd
import datetime
from celery import Celery
from report_manager.apps import projectCreation
from graphdb_connector import connector

celery_app = Celery('create_new_project', broker='redis://localhost:6379')


@celery_app.task
def create_new_project(identifier, data):
    print("Holaaaaam")
    driver = connector.getGraphDatabaseConnectionConfiguration()
    result = projectCreation.create_new_project(driver, identifier, pd.read_json(data))
    
    return result
