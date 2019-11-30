#!/bin/bash

echo "$USER"
service neo4j version
service neo4j start
sleep 60
cat /var/log/neo4j/neo4j.log

git clone https://github.com/MannLabs/CKG.git

echo "Setting up the config files"
python3 /CKG/setup_config_files.py

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
./etc/redis/bin/redis-server

echo "Initiating queue celery"
cd /CKG/src
celery -A worker worker --loglevel=DEBUG --concurrency=3 -E

if [ -e /debug1 ]; then
  echo "Running app in debug mode!"
  python3 /CKG/src/report_manager/index.py
else
  echo "Running app in production mode!"
  nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini
fi
