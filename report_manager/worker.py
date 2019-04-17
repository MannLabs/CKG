import os
import time
import datetime
from celery import Celery


celery_app = Celery('hello', broker='redis://localhost:6379')


@celery_app.task
def hello():
    time.sleep(10)
    with open ('hellos.txt', 'a') as hellofile:
        hellofile.write('Hello {}\n'.format(datetime.datetime.now()))
