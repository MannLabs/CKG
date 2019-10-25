#Download base image ubuntu 16.04
FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
ENV LC_CTYPE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV R_BASE_VERSION 3.6.1

MAINTAINER Alberto Santos "alberto.santos@cpr.ku.dk"

USER root

RUN apt-get update && \
    apt-get -yq dist-upgrade && \
    apt-get install -yq --no-install-recommends && \
    apt-get install -yq locales && \
    apt-get install -yq wget && \
    apt-get install -yq unzip && \
    apt-get install -yq python3.6 python3-pip python3-setuptools python3-dev libxml2 libxml2-dev zlib1g-dev && \
    apt-get install -yq nginx uwsgi uwsgi-plugin-python3 && \
    rm -rf /var/lib/apt/lists/*

# Set the locale
RUN locale-gen en_US.UTF-8

# gpg key for cran updates
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    gpg -a --export E084DAB9 > cran.asc && \
    apt-key add cran.asc

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9   

RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/" > /etc/apt/sources.list.d/cran.list

# JAVA
RUN mkdir -p /usr/local/oracle-jre8-installer-local

ADD /resources/jre-8u221-linux-x64.tar.gz /usr/local/oracle-jre8-installer-local

RUN update-alternatives --install "/usr/bin/java" "java" "/usr/local/oracle-jre8-installer-local/jre1.8.0_221/bin/java" 1500 && \
    update-alternatives --install "/usr/bin/javac" "javac" "/usr/local/oracle-jre8-installer-local/jre1.8.0_221/bin/javaws" 1500 

# NEO4J
RUN wget -O - http://debian.neo4j.org/neotechnology.gpg.key | apt-key add - && \
    echo "deb http://debian.neo4j.org/repo stable/" > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install -yq neo4j=1:3.5.8

## Setup initial user Neo4j
RUN rm -f /var/lib/neo4j/data/dbms/auth && \
    neo4j-admin set-initial-password "bioinfo1112"

## Install algorithms Neo4j
RUN wget -P /var/lib/neo4j/plugins https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/neo4j-graph-algorithms-3.5.8.0-standalone.zip && \
    unzip /var/lib/neo4j/plugins/neo4j-graph-algorithms-3.5.8.0-standalone.zip -d /var/lib/neo4j/plugins && \
    rm /var/lib/neo4j/plugins/neo4j-graph-algorithms-3.5.8.0-standalone.zip

RUN wget -P /var/lib/neo4j/plugins https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/3.5.0.4/apoc-3.5.0.4-all.jar 

RUN ls -lrth /var/lib/neo4j/plugins

## Test the service Neo4j
RUN service neo4j start && \
    cat /var/log/neo4j/neo4j.log && \
    sleep 30 && \
    service neo4j stop

## Change configuration
COPY /resources/neo4j_db/neo4j.conf  /etc/neo4j/.

## Load backup with Clinical Knowledge Graph
COPY /resources/neo4j_db/backups /var/lib/neo4j/data/backups
RUN mkdir -p /var/lib/neo4j/data/databases/graph.db
RUN ls -lrth /var/lib/neo4j/data/
RUN neo4j-admin load --from=/var/lib/neo4j/data/backups/graph.db/2019_1909.dump --database=graph.db --force


## Remove dump file
RUN echo "Done with restoring backup, removing backup folder"
RUN rm -rf /var/lib/neo4j/data/backups 

RUN ls -lrth  /var/lib/neo4j/data/databases
RUN [ -e  /var/lib/neo4j/data/databases/store_lock ] && rm /var/lib/neo4j/data/databases/store_lock

# R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
        littler \
        r-cran-littler \
		r-base=${R_BASE_VERSION}* \
		r-base-dev=${R_BASE_VERSION}* \
		r-recommended=${R_BASE_VERSION}* && \
        echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site
## Install packages
ADD ./R_packages.R /R_packages.R
RUN Rscript R_packages.R

# Python
## Copy Requirements
ADD ./requirements.txt /requirements.txt

## Install Python libraries
RUN pip3 install --ignore-installed -r requirements.txt
RUN mkdir /CKG
ADD . /CKG/
ENV PYTHONPATH "${PYTHONPATH}:/CKG"

# JupyterHub
RUN apt-get -y install npm nodejs && \
    npm install -g configurable-http-proxy
    
RUN pip3 install jupyterhub && \
    pip3 install --upgrade notebook

RUN apt-get remove -y python-pip curl && \
         rm -rf /var/lib/apt/lists/

## Add a user without password in JupyterHub
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/adminhub --gecos "User" adminhub && \
    echo "adminhub:adminhub" | chpasswd

RUN mkdir /etc/jupyterhub
COPY /resources/jupyterhub.py /etc/jupyterhub/.


# NGINX and UWSGI

## Copy configuration file
COPY /resources/nginx.conf /etc/nginx/nginx.conf

RUN adduser --disabled-password --gecos '' nginx\
  && chown -R nginx:nginx /CKG \
  && chmod 777 /run/ -R \
  && chmod 777 /root/ -R

## Copy the base uWSGI ini file
COPY /resources/uwsgi.ini /etc/uwsgi/apps-available/uwsgi.ini
RUN ln -s /etc/uwsgi/apps-available/uwsgi.ini /etc/uwsgi/apps-enabled/uwsgi.ini

## Create log directory
RUN mkdir -p /var/log/uwsgi

# Expose ports (HTTP Neo4j, Bolt Neo4j, jupyterHub, CKG)
EXPOSE 7474 7687 8090 8050

ENTRYPOINT [ "/bin/bash", "/CKG/docker_entrypoint.sh"]
