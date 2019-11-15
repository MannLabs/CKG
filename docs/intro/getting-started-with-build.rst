Getting Started with the CKG Build
==================================

Setting up the Clinical Knowledge Graph is easy.

Assuming you have Python 3 already installed and added to ''PATH'', you can choose to create a virtual environment where all the packages with the specific versions will be installed. To do so, use Virtualenv.

Create a virtual environment
----------------------------

Virtualenv is not installed by default on Macbook machines. To install it, run:

.. prompt:: bash $

	pip install virtualenv

To create a new virtual environment using a costum version of Python, follow the steps:

1. Make  note of the full file path to the Python version you would like to use inside the virtual environment.

2. Navigate to the directory where you would like your virtual environment to be (e.g. user's root).

3. Create the virtual environment at the same time you specify the version of Python you wish to use. ''env_name'' is the name of the virtual environment and can be set to anything you like.

.. prompt:: bash $

	virtualenv -p /path/to/python env_name

4. Activate the virtual environment by running:

.. prompt:: bash $

	source env_name/bin/activate

After this, the name of the virtual environment will now appear on the left of the prompt:

(env_name) username$

If you are finished working in the virtual environment for the moment, you can deactivate it by running:

.. prompt:: bash $

	deactivate


Install python modules with pip
-------------------------------

All the Python modules neccessary to run the Clinical Knowledge graph can be found in ''requirements.txt''.
To install all the packages required, run:

.. prompt:: bash $

	pip install -r requirements.txt



Add CKG to *.bashrc*
--------------------

In order run the the Clinical Knowledge Graph, add the path to the code to your .bashrc (or .bash_profile):

1. Open the .bashrc file.

2. Add the following lines to the file and save it:

PYTHONPATH="${PYTHONPATH}:/path/to/folder/CKG/src/"
export PYTHONPATH

Notice that the path should always finish with ''/CKG/src/''.


3. Reload .bashrc:

.. prompt:: bash $

	source ~/.bashrc



Build Neo4j graph database
--------------------------

In order to start building the Clinical Knowledge Graph database, you will have to create the appropriate directory architecture:

.. prompt:: bash $

	cd CKG/
	python setup_CKG.py
	python setup_config_files.py

This will automatically create the ''data'' folder and all subfolders, as well as setup the configuration for the log files where
all errors and warnings related to the code will be written to.

Regarding the ''data'', most of the biomedical databases and ontology files will automatically be downloaded during building
of the database. However, the following have to be downloaded manually.

- 
-
-

After download, move the files to their respective folders:

-
-
-


To build the graph database, run ''builder.py'':

.. prompt:: bash $
	
	cd src/graphdb_builder/builder
	python builder.py -b full -u neo4j


This action will take aproximately 10 hours but depending on a multitude of factors, it can take up to 16 hours.

Alternatively, you can use the available dump file and load the graph database contained in it:

.. prompt:: bash $

	cd /path/to/neo4jDatabases/database-identifier/installation-x.x.x/
	mkdir backups
	mkdir backups/graph.db
	cp 2019-11-04.dump backups/graph.db/.

After copying the dump file to backups/graph.db/, make sure the graph database e shut down and run:

.. prompt:: bash $
	
	bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force

In some systems you might have to run this as root:

.. prompt:: bash $
	
	sudo bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force
	sudo chown -R username data/databases/graph.db/

Once you are done, restart the database and you are good to go!


