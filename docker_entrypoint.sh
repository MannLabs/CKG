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

if [[ -e $EXEC_MODE ]]; then
  echo "Mode:: $EXEC_MODE"
  if [ $EXEC_MODE == "load" ]; then
    if [[ -z $DUMP_PATH ]]; then
      echo "Loading database dump: $DUMP_PATH"
      sudo -u neo4j neo4j-admin load --from=$DUMP_PATH --database=graph.db --force
    fi
  elif [ $EXEC_MODE == "minimal" ]; then
    echo "Building CKG graph database after loading database dump"
    cd src/graphdb_builder/builder
    python3 builder.py -b minimal -u ckg
  elif [ $EXEC_MODE == "build" ]; then
    echo "Setting up the config files"
    python3 setup_CKG.py
    python3 setup_config_files.py
    echo "Building CKG full graph database"
    cd src/graphdb_builder/builder
    python3 builder.py -b full -u ckg
  fi
fi

echo "Initiating celery queues"
cd src/report_manager
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E &
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E -Q compute &


echo "Initiating CKG app"
nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini --uid 1500 --gid nginx
