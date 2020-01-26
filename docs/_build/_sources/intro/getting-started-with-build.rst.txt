Getting Started with the CKG Build
===================================

Setting up the Clinical Knowledge Graph is straightforward.

Assuming you have **Python 3.6** already installed and added to ``PATH``, you can choose to create a virtual environment where all the packages with the specific versions will be installed. To do so, use Virtualenv.

To check which Python version is currently installed:

.. prompt:: bash $
	
	python3 --version

And where this Python version is:

.. prompt:: bash $

	which python3

If this does not correspond to the correct Python version you want to run, you can create a shell alias in the bash file:

1. Open the bash file:

.. prompt:: bash $
	
	vi ~/.bash_profile

2. Add at the end of the file:

.. promtp:: python
	
	alias python3="/path/to/correct/python3.6"

3. Save and close the bash file

4. Make the alias available in the current session:

.. prompt:: bash $
	
	source ~/.bash_profile

.. note:: If you don't have **Python 3.6** installed, go to ``pyhton.org`` and download the Python 3.6 version appropriate for your machine, and run the installer package. Python should be installed in ``/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6`` and also found in ``/usr/local/bin/python3.6``.



Create a virtual environment
-----------------------------

Virtualenv is not installed by default on Macbook machines. To install it, run:

.. prompt:: bash $

	python3 -m pip install virtualenv

To create a new virtual environment using a costum version of Python, follow the steps:

1. Take note of the full path to the Python version you would like to use inside the virtual environment.

2. Navigate to the directory where you would like your virtual environment to be (e.g. user's root).

3. Create the virtual environment at the same time you specify the version of Python you wish to use. ``env_name`` is the name of the virtual environment and can be set to anything you like.

.. prompt:: bash $

	virtualenv -p /path/to/python env_name

4. Activate the virtual environment by running:

.. prompt:: bash $

	source env_name/bin/activate

After this, the name of the virtual environment will now appear on the left of the prompt:

.. prompt:: bash

	(env_name) username$

If you are finished working in the virtual environment for the moment, you can deactivate it by running:

.. prompt:: bash $

	deactivate


Setting up the Clinical Knowledge Graph
-----------------------------------------

The first step in setting up the CKG, is to obtain the complete code by clone the GitHub repository:

.. prompt:: bash $

	git clone https://github.com/MannLabs/CKG.git

Once this is finished, you can find all the Python modules neccessary to run the Clinical Knowledge graph in ``requirements.txt``.
To install all the packages required, simply run:

.. prompt:: bash $

	pip install -r requirements.txt

.. warning:: Make sure the virtual environment previously created is active before installing ``requirements.txt``.

Now that all the packages are correctly installed, you will have to create the appropriate directory architecture within the local copy of the cloned repository:

.. prompt:: bash $

	cd CKG/
	python setup_CKG.py
	python setup_config_files.py

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where
all errors and warnings related to the code will be written to.


Add CKG to *.bashrc*
---------------------

In order run the the Clinical Knowledge Graph, add the path to the code to your ``.bashrc`` (or ``.bash_profile``):

1. Open the .bashrc file.

2. Add the following lines to the file and save it:

.. prompt:: bash
	
	PYTHONPATH="${PYTHONPATH}:/path/to/folder/CKG/src/"
	export PYTHONPATH

Notice that the path should always finish with "/CKG/src/".


3. To reload the bash file, first deactivate the virtual environment, reload ~/.bashrc, and activate the virtual environment again:

.. prompt:: bash $
	
	deactivate
	source ~/.bashrc
	source env_name/bin/activate


Build Neo4j graph database
---------------------------

The building of the CKG database is thoroughly automated. most of the biomedical databases and ontology files will automatically be downloaded during building
of the database. However, the following have to be downloaded manually.

- `PhosphoSitePlus <https://www.phosphosite.org/staticDownloads>`_: *Acetylation_site_dataset.gz*, *Disease-associated_sites.gz*, *Kinase_Substrate_Dataset.gz*, *Methylation_site_dataset.gz*, *O-GalNAc_site_dataset.gz*, *O-GlcNAc_site_dataset.gz*, *Phosphorylation_site_dataset.gz*, *Regulatory_sites.gz*, *Sumoylation_site_dataset.gz* and *Ubiquitination_site_dataset.gz*.

- `DrugBank <https://www.drugbank.ca/releases/latest>`_: *All drugs* and *DrugBank Vocabulary*.

- `SNOMED-CT <https://www.nlm.nih.gov/healthit/snomedct/us_edition.html>`_: *Download SNOMED CT to ICD-10-CM Mapping Resources*.


.. warning:: These three databases require login and authentication. To sign up go to `PSP Sign up <https://www.phosphosite.org/signUpAction>`_, `DrugBank Sign up <https://www.drugbank.ca/public_users/sign_up>`_ and `SNOMED-CT Sign up <https://uts.nlm.nih.gov/license.html>`_. In the case of SNOMED-CT, the UMLS license can take several business days.

After download, move the files to their respective folders:

- PhosphoSitePlus: ``CKG/data/databases/PhosphoSitePlus``
- DrugBank: ``CKG/data/databases/DrugBank``
- SNOMED-CT: ``CKG/data/ontologies/SNOMED-CT``

.. note:: If the respective database folder is not created, please do it manually.

The last step is to build the database, which can be done using the ``builder.py`` module or a ``dump file``.


From builder.py
^^^^^^^^^^^^^^^^^^

To build the graph database, run ``builder.py``:

.. prompt:: bash $
	
	cd src/graphdb_builder/builder
	python builder.py -b full -u neo4j

.. warning:: Before running ``builder.py``, please make sure your Neo4j graph is running. The builder will fail otherwise.

This action will take aproximately 10 hours but depending on a multitude of factors, it can take up to 16 hours.


From a backup dump file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can use the available dump file and load the graph database contained in it:

.. prompt:: bash $

	cd /path/to/neo4jDatabases/database-identifier/installation-x.x.x/
	mkdir backups
	mkdir backups/graph.db
	cp 2019-11-04.dump backups/graph.db/.

After copying the dump file to backups/graph.db/, make sure the graph database is shutdown and run:

.. prompt:: bash $
	
	bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force

.. warning:: Make sure the dump file naming in the command above, matches the one provided to you.

In some systems you might have to run this as root:

.. prompt:: bash $
	
	sudo bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force
	sudo chown -R username data/databases/graph.db/

Once you are done, restart the database and you are good to go!








