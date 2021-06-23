#!/bin/bash

##################################################################################
# --------------------------------------------------------------------------------
# [Alberto Santos] -- June 2021 -- Center for Health Data Science (Copenhagen, DK)
#
#          Clinical Knowledge Graph Ubuntu Installation
#          --------------------------------------------
#          This script will install CKG's full framework:
#          CKG's python library and app, Neo4j Database and 
#          JupyterHub. Further, it will setup a production
#          environment to run CKG app.
#          
#          To install:
#          > ./ubuntu_install.sh
#
#          After installing, you can use `ubuntu_run.sh` to start
#          CKG's framework:
#          > ./ubuntu_run.sh
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
##################################################################################

# Globals
R_BASE_VERSION=3.6.1
PYTHON_VERSION=3.7.9
NEO4J_VERSION=4.2.3
BASEDIR=${PWD}

echo "Installing essential tools"
apt-get update && \
    apt-get -yq dist-upgrade && \
    apt-get install -yq --no-install-recommends && \
    apt-get install -yq apt-utils software-properties-common && \
    apt-get install -yq locales && \
    apt-get install -yq wget && \
    apt-get install -yq unzip && \
    apt-get install -yq build-essential sqlite3 libsqlite3-dev libxml2 libxml2-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libcurl4-openssl-dev && \
    apt-get install -yq nginx && \
    apt-get install -yq redis-server && \
    apt-get install -y dos2unix && \
    apt-get install -yq git && \
    apt-get install -y sudo && \
    apt-get install -y net-tools &&\
    rm -rf /var/lib/apt/lists/*

echo "Creating CKG Unix users and groups"
groupadd ckg_group && \
adduser --quiet --disabled-password --shell /bin/bash --home /home/adminhub --gecos "User" adminhub && \
echo "adminhub:adminhub" | chpasswd && \
usermod -a -G ckg_group adminhub && \
adduser --quiet --disabled-password --shell /bin/bash --home /home/ckguser --gecos "User" ckguser && \
echo "ckguser:ckguser" | chpasswd && \
usermod -a -G ckg_group ckguser && \
adduser --disabled-password --gecos '' --uid 1500 nginx && \
usermod -a -G ckg_group nginx

echo "Python installation"
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
./configure
make altinstall
make install
## pip upgrade
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip3 install --upgrade pip
pip3 install setuptools


echo "Installing Java"
cd ${BASEDIR}
# gpg key for cran updates
gpg --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    gpg -a --export E084DAB9 > cran.asc && \
    apt-key add cran.asc
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9   
echo "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/" > /etc/apt/sources.list.d/cran.list
# Installation openJDK 11
add-apt-repository ppa:openjdk-r/ppa
apt-get update
apt-get install -yq openjdk-11-jdk
java -version
javac -version 

echo "Installing Neo4j Database"
# NEO4J 4.2.3
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo "deb [trusted=yes] https://debian.neo4j.com stable 4.2" > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install -yq neo4j=1:4.2.3

## Setup initial user Neo4j
rm -f /var/lib/neo4j/data/dbms/auth && \
    neo4j-admin set-initial-password "NeO4J"

## Install graph data science library and APOC
wget -P /var/lib/neo4j/plugins https://github.com/neo4j/graph-data-science/releases/download/1.5.1/neo4j-graph-data-science-1.5.1.jar
wget -P /var/lib/neo4j/plugins https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.2.0.4/apoc-4.2.0.4-all.jar

## Change configuration
cp resources/neo4j_db/neo4j.conf  /etc/neo4j/.
dos2unix /etc/neo4j/neo4j.conf

## Test the service Neo4j
service neo4j start && \
    sleep 60 && \
    ls -lrth /var/log && \
    service neo4j stop

echo "Restoring CKG's Database dump file"
# Load backup with Clinical Knowledge Graph
mkdir -p /var/lib/neo4j/data/backup
cp resources/neo4j_db/ckg_190521_neo4j_4.2.3.dump /var/lib/neo4j/data/backup/.
mkdir -p /var/lib/neo4j/data/databases/graph.db
sudo -u neo4j neo4j-admin load --from=/var/lib/neo4j/data/backup/ckg_190521_neo4j_4.2.3.dump --database=graph.db --force

# # Remove dump file
echo "Done with restoring backup, removing backup folder"
rm -rf /var/lib/neo4j/data/backup

#RUN ls -lrth  /var/lib/neo4j/data/databases
[ -e  /var/lib/neo4j/data/databases/store_lock ] && rm /var/lib/neo4j/data/databases/store_lock
#RUN [ -e  /var/lib/neo4j/data/databases/store_lock ] && rm /var/lib/neo4j/data/databases/store_lock


echo "Installing R and R packages"
#R
apt-get update && \
   apt-get install -y --no-install-recommends \ 
   littler \
   r-cran-littler \
   r-base=${R_BASE_VERSION}* \
   r-base-dev=${R_BASE_VERSION}* \
   r-recommended=${R_BASE_VERSION}* && \
   echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site
    

echo "Installing CKG app"
# CKG Python library
## Install Python libraries
python3 -m pip install --ignore-installed -r requirements.txt
###Creating CKG directory and setting up CKG
chown -R nginx ckg
echo "export ${PYTHONPATH}:/CKG" >> ~/.profile
### Installation
python3 ckg/init.py
#### Directory ownership 
chown -R nginx .
chgrp -R ckg_group .


echo "Installing JupyterHub"
cd ${BASEDIR}
# JupyterHub
apt-get -y install npm nodejs && \
    npm install -g configurable-http-proxy
    
pip3 install jupyterhub

mkdir /etc/jupyterhub
cp resources/jupyterhub.py /etc/jupyterhub/.
cp -r ckg/notebooks /home/adminhub/.
cp -r ckg/notebooks /home/ckguser/.
chown -R adminhub /home/adminhub/notebooks
chgrp -R adminhub /home/adminhub/notebooks
chown -R ckguser /home/ckguser/notebooks
chgrp -R ckguser /home/ckguser/notebooks

ls -alrth /home/ckguser
ls -alrth /home/ckguser/notebooks


echo "Installing and setting up NGINX and UWSGI"
# NGINX and UWSGI
## Copy configuration file
cp resources/nginx.conf /etc/nginx/.

chmod 777 /run/ -R && \
    chmod 777 /root/ -R

## Install uWSGI
pip3 install uwsgi

## Copy the base uWSGI ini file
cp resources/uwsgi.ini /etc/uwsgi/apps-available/uwsgi.ini
cp resources/uwsgi.ini /etc/uwsgi/apps-enabled/uwsgi.ini
## Create log directory
mkdir -p /var/log/uwsgi
