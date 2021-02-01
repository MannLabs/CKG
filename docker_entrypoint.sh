#!/bin/bash
echo "Entry point to CKG Docker"
echo $DUMP_PATH
echo $EXEC_MODE

cd /CKG

echo "Starting Neo4j"
service neo4j start &
service neo4j status

#wget --quiet --tries=10 --waitretry=32 -O /dev/null http://localhost:7474
while ! [[ `wget -S --spider http://localhost:7474  2>&1 | grep 'HTTP/1.1 200 OK'` ]]; do
echo "Database not ready"
sleep 60
done

echo "Database ready"

echo "Creating Test user in the database"
python3 src/graphdb_builder/builder/create_user.py -u test_user -d test_user -n test -e test@ckg.com -a test


echo "Running jupyterHub"
jupyterhub -f /etc/jupyterhub/jupyterhub.py --no-ssl &

echo "Running redis-server"
service redis-server start

echo "Initiating celery queues"
cd src/report_manager
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E &
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E -Q update &
celery -A worker worker --loglevel=DEBUG --concurrency=3 --uid=1500 --gid=nginx -E -Q compute &


echo "Initiating CKG app"
nginx && uwsgi --ini /etc/uwsgi/apps-enabled/uwsgi.ini --uid 1500 --gid nginx
