#!/bin/bash

##################################################################################
# --------------------------------------------------------------------------------
# [Alberto Santos] -- June 2021 -- Center for Health Data Science (Copenhagen, DK)
#
#          Clinical Knowledge Graph framework
#          ----------------------------------
#          This script runs CKG's full framework:
#          CKG's python library and app, Neo4j Database and 
#          JupyterHub in a production environment.
#          
#          To run:
#          > ./ubuntu_run.sh
#
#          After running, the following ports will be accessible:
#           - localhost:7474 - Neo4j Database browser
#           - localhost:7687 - Neo4j Database bolt (programmatic access)
#           - localhost:8090 - JupyterHub server
#           - loalhost:8050 - CKG app
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
##################################################################################

BASEDIR=${PWD}

echo "Starting Neo4j"
service neo4j start &
service neo4j status

while ! [[ `wget -S --spider http://localhost:7474  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; do
echo "Database not ready"
sleep 60
done
echo "Database ready"

echo "Creating Test user in the database"
python3 ckg/graphdb_builder/builder/create_user.py -u test_user -d test_user -n test -e test@ckg.com -a test

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
service redis-server start

echo "Running celery queues"
cd ckg/report_manager
celery -A ckg.report_manager.worker worker --loglevel=INFO --concurrency=1 -E -Q creation &
celery -A ckg.report_manager.worker worker --loglevel=INFO --concurrency=3 -E -Q compute &
celery -A ckg.report_manager.worker worker --loglevel=INFO --concurrency=1 -E -Q update &

cd ${BASEDIR}
echo "Initiating CKG app"
nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini --uid 1500 --gid nginx