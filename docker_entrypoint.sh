#!/bin/bash

service neo4j start

cat /var/log/neo4j/neo4j.log

sleep 60

echo "Setting up the config files"
python3 /CKG/setup_config_files.py

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

if [ -e /debug1 ]; then
  echo "Running app in debug mode!"
  python3 /CKG/report_manager/index.py
else
  echo "Running app in production mode!"
  nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini
fi
