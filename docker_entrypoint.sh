#!/bin/bash
echo "Entry point to CKG Docker"
echo $DUMP_PATH
echo $EXEC_MODE

cd /CKG

echo "Starting Neo4j"
service neo4j start
service neo4j status

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
service redis-server start

echo "Initiating celery queues"
cd src/report_manager
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E &
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E -Q compute &


echo "Initiating CKG app"
nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini --uid 1500 --gid nginx
