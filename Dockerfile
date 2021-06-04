#Download base image ubuntu
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
ENV LC_CTYPE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV R_BASE_VERSION 3.6.1

LABEL maintainer "Alberto Santos alberto.santos@sund.ku.dk"

USER root

RUN apt-get update && \
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


# User management
RUN groupadd ckg_group && \
    adduser --quiet --disabled-password --shell /bin/bash --home /home/adminhub --gecos "User" adminhub && \
    echo "adminhub:adminhub" | chpasswd && \
    usermod -a -G ckg_group adminhub && \
    adduser --quiet --disabled-password --shell /bin/bash --home /home/ckguser --gecos "User" ckguser && \
    echo "ckguser:ckguser" | chpasswd && \
    usermod -a -G ckg_group ckguser && \
    adduser --disabled-password --gecos '' --uid 1500 nginx && \
    usermod -a -G ckg_group nginx


RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
RUN tar -xzf Python-3.7.9.tgz
WORKDIR /Python-3.7.9
RUN ./configure
RUN make altinstall
RUN make install
## pip upgrade
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install setuptools

WORKDIR /

# Set the locale
RUN locale-gen en_US.UTF-8

# gpg key for cran updates
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    gpg -a --export E084DAB9 > cran.asc && \
    apt-key add cran.asc

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9   

RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/" > /etc/apt/sources.list.d/cran.list

# Installation openJDK 11
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -yq openjdk-11-jdk
RUN java -version
RUN javac -version 

# NEO4J 4.2.3
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo "deb [trusted=yes] https://debian.neo4j.com stable 4.2" > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install -yq neo4j=1:4.2.3

## Setup initial user Neo4j
RUN rm -f /var/lib/neo4j/data/dbms/auth && \
    neo4j-admin set-initial-password "NeO4J"

## Install graph data science library and APOC
RUN wget -P /var/lib/neo4j/plugins https://github.com/neo4j/graph-data-science/releases/download/1.5.1/neo4j-graph-data-science-1.5.1.jar
RUN wget -P /var/lib/neo4j/plugins https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.2.0.4/apoc-4.2.0.4-all.jar

## Change configuration
COPY /resources/neo4j_db/neo4j.conf  /etc/neo4j/.
RUN dos2unix /etc/neo4j/neo4j.conf

## Test the service Neo4j
RUN service neo4j start && \
    sleep 60 && \
    ls -lrth /var/log && \
    service neo4j stop

# Load backup with Clinical Knowledge Graph
RUN mkdir -p /var/lib/neo4j/data/backup
#RUN wget -O /var/lib/neo4j/data/backup/latest.dump https://data.mendeley.com/public-files/datasets/mrcf7f4tc2/files/e454e520-2384-49ee-ac41-d4fbfa15c065/file_downloaded
COPY /resources/neo4j_db/ckg_190521_neo4j_4.2.3.dump /var/lib/neo4j/data/backup/.
RUN mkdir -p /var/lib/neo4j/data/databases/graph.db
RUN sudo -u neo4j neo4j-admin load --from=/var/lib/neo4j/data/backup/ckg_190521_neo4j_4.2.3.dump --database=graph.db --force

# # Remove dump file
RUN echo "Done with restoring backup, removing backup folder"
RUN rm -rf /var/lib/neo4j/data/backup

#RUN ls -lrth  /var/lib/neo4j/data/databases
RUN [ -e  /var/lib/neo4j/data/databases/store_lock ] && rm /var/lib/neo4j/data/databases/store_lock
#RUN [ -e  /var/lib/neo4j/data/databases/store_lock ] && rm /var/lib/neo4j/data/databases/store_lock

#R
RUN apt-get update && \
   apt-get install -y --no-install-recommends \ 
   littler \
   r-cran-littler \
   r-base=${R_BASE_VERSION}* \
   r-base-dev=${R_BASE_VERSION}* \
   r-recommended=${R_BASE_VERSION}* && \
   echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site
    

# CKG Python library
COPY ./requirements.txt .
## Install Python libraries
RUN python3 -m pip install --ignore-installed -r requirements.txt
###Creating CKG directory and setting up CKG
RUN mkdir /CKG
COPY --chown=nginx ckg /CKG/ckg
COPY docker_entrypoint.sh /CKG/.
ENV PYTHONPATH "${PYTHONPATH}:/CKG"
RUN ls -lrth /CKG
### Installation
WORKDIR /CKG
RUN python3 ckg/init.py
#### Directory ownership 
RUN chown -R nginx .
RUN chgrp -R ckg_group .

WORKDIR /

# JupyterHub
RUN apt-get -y install npm nodejs && \
    npm install -g configurable-http-proxy
    
RUN pip3 install jupyterhub

RUN mkdir /etc/jupyterhub
COPY /resources/jupyterhub.py /etc/jupyterhub/.
RUN cp -r /CKG/ckg/notebooks /home/adminhub/.
RUN cp -r /CKG/ckg/notebooks /home/ckguser/.
RUN chown -R adminhub /home/adminhub/notebooks
RUN chgrp -R adminhub /home/adminhub/notebooks
RUN chown -R ckguser /home/ckguser/notebooks
RUN chgrp -R ckguser /home/ckguser/notebooks

RUN ls -alrth /home/ckguser
RUN ls -alrth /home/ckguser/notebooks

# NGINX and UWSGI
## Copy configuration file
COPY /resources/nginx.conf /etc/nginx/.

RUN chmod 777 /run/ -R && \
    chmod 777 /root/ -R

## Install uWSGI
RUN pip3 install uwsgi

## Copy the base uWSGI ini file
COPY /resources/uwsgi.ini /etc/uwsgi/apps-available/uwsgi.ini
COPY /resources/uwsgi.ini /etc/uwsgi/apps-enabled/uwsgi.ini

## Create log directory
RUN mkdir -p /var/log/uwsgi

# Remove apt cache to make the image smaller
RUN rm -rf /var/lib/apt/lists/*

RUN chmod +x /CKG/docker_entrypoint.sh && ln -s /CKG/docker_entrypoint.sh
RUN dos2unix /CKG/docker_entrypoint.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

RUN ls -alrth /
RUN ls -alrth /CKG
RUN ls -alrth /CKG/ckg/notebooks
RUN pip3 freeze
# Expose ports (HTTP Neo4j, Bolt Neo4j, jupyterHub, CKG prod, CKG dev, Redis)
EXPOSE 7474 7687 8090 8050 5000 6379

ENTRYPOINT ["/bin/bash", "/CKG/docker_entrypoint.sh"]
