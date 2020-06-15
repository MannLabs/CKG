#!/bin/bash
echo "Entry point to CKG Docker"
echo $DUMP_PATH
echo $EXEC_MODE

if [[ -e $EXEC_MODE ]]; then
  echo "Mode:: $EXEC_MODE"
  if [ $EXEC_MODE == "load" ]; then
    if [[ -z $DUMP_PATH ]]; then
      echo "Loading database dump: $DUMP_PATH"
      sudo -u neo4j neo4j-admin load --from=$DUMP_PATH --database=graph.db --force
    fi
  elif [ $EXEC_MODE == "minimal" ]; then
    echo "Setting up the config files"
    python3 /CKG/setup_CKG.py
    python3 /CKG/setup_config_files.py
    echo "Building CKG graph database after loading database dump"
    cd /CKG/src/graphdb_builder/builder
    python3 builder.py -b minimal -u ckg
  elif [ $EXEC_MODE == "build" ]; then
    echo "Setting up the config files"
    python3 /CKG/setup_CKG.py
    python3 /CKG/setup_config_files.py
    echo "Building CKG full graph database"
    cd /CKG/src/graphdb_builder/builder
    python3 builder.py -b full -u ckg
  fi
fi

echo "Starting Neo4j"
service neo4j version
service neo4j start
cat /var/log/neo4j/neo4j.log
sleep 60

echo "Changing user to CKG"
su - ckg_user

echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
service redis-server start

echo "Initiating queue celery"
cd /CKG/src/report_manager
celery -A worker worker --loglevel=DEBUG --concurrency=3 -E &
celery -A worker worker --loglevel=DEBUG --concurrency=3 -E -Q compute &


if [[ -e $SERVER_MODE ]]; then
  if [ $SERVER_MODE == "debug"]; then
    echo "Running app in debug mode!"
    python3 /CKG/src/report_manager/index.py
  fi
else
  echo "Running app in production mode!"
  nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini
fi
