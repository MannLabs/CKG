
Getting started with Docker **(Testing)**
============================================

In this section we describe how to set up the Clinical Knowledge Graph from a Docker container.
This container will install all the requirements needed, download source databases and build the CKG graph database, and open 5 ports through which to interact with the CKG.

To run the Docker, simply:

1. Allocate resources:

The docker build requires more resources (memory and disk) than the ones set as default. Make sure to allocate at least 8Gb memory and at least 60Gb of Disk space. To change these settings: Docker Preferences -> Resources.


2. Build the docker

.. code-block:: bash
	
	$ cd CKG/
	$ docker build -t docker-ckg:latest .


3. Make sure to download manually the licensed databases (:ref:`Build Neo4j graph database`)


4. Run the docker

.. code-block:: bash

	$ docker run -d --name ckgapp -d -v log:/CKG/log -v data:/CKG/data -e EXEC_MODE="minimal" --restart=always -p 8050:8050 -p 7470:7474 -p 8090:8090 -p 7680:7687 -p 6379:6379 docker-ckg:latest


.. note:: Be aware, this requires Docker to be previously installed.
