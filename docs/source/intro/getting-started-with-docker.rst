
Getting started with Docker **(Testing)**
============================================

In this section we describe how to set up the Clinical Knowledge Graph from a Docker container.
This container will install all the requirements needed, download source databases and build the CKG graph database, and open 5 ports through which to interact with the CKG.

A requirement to run the Docker is **Java SE Runtime Environment**.
Please go to ``https://www.oracle.com/java/technologies/javase-jre8-downloads.html`` and download the ``jre-8u221-linux-x64.tar.gz`` file. Once downloaded, place it in ``CKG/resources/``.

To run the Docker, simply:

1. Build the docker

.. code-block:: bash
	
	$ cd CKG/
	$ docker build -t docker-ckg:latest .


2. Run the docker

.. code-block:: bash

	$ docker run -d --name ckgapp -d -v log:/CKG/log -v data:/CKG/data -e EXEC_MODE="minimal" --restart=always -p 8050:8050 -p 7470:7474 -p 8090:8090 -p 7680:7687 -p 6379:6379 docker-ckg:latest


.. note:: Be aware, this requires Docker to be previously installed.