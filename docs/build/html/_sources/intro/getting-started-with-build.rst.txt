Getting Started with the CKG Build
===================================

Setting up the Clinical Knowledge Graph is straightforward.
Assuming you have **Python 3.6** already installed and added to ``PATH``, you can choose to create a virtual environment where all the packages with the specific versions will be installed. To do so, use Virtualenv.

To check which Python version is currently installed:

.. code-block:: bash
	
	$ python3.6 --version

And where this Python version is:

.. code-block:: bash

	$ which python3.6

If this does not correspond to the correct Python version you want to run, you can create a shell alias in the bash file:

1. Open the bash file:

.. code-block:: bash
	
	$ vi ~/.bash_profile

#. Add at the end of the file:

.. code-block:: bash
	
	alias python3.6="/path/to/correct/python3.6"

#. Save and close the bash file

#. Make the alias available in the current session:

.. code-block:: bash
	
	$ source ~/.bash_profile

.. note:: If you don't have **Python 3.6** installed, `download <https://www.python.org/>`__ the Python 3.6 version appropriate for your machine, and run the installer package. Python should be installed in ``/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6`` and also found in ``/usr/local/bin/python3.6``.



Create a virtual environment
-----------------------------

Virtualenv is not installed by default on Macbook machines. To install it, run:

.. code-block:: bash

	$ python3 -m pip install virtualenv

To create a new virtual environment using a costum version of Python, follow the steps:

1. Take note of the full path to the Python version you would like to use inside the virtual environment.

#. Navigate to the directory where you would like your virtual environment to be (e.g. user's root).

#. Create the virtual environment at the same time you specify the version of Python you wish to use. ``env_name`` is the name of the virtual environment and can be set to anything you like.

.. code-block:: bash

	$ virtualenv -p /path/to/python env_name

#. Activate the virtual environment by running:

.. code-block:: bash

	$ source path/to/env_name/bin/activate

After this, the name of the virtual environment will now appear on the left of the prompt:

.. code-block:: bash

	(env_name) username$

If you are finished working in the virtual environment for the moment, you can deactivate it by running:

.. code-block:: bash

	$ deactivate


Setting up the Clinical Knowledge Graph
-----------------------------------------

The first step in setting up the CKG, is to obtain the complete code by clone the GitHub repository:

.. code-block:: bash
	
	$ git clone https://github.com/MannLabs/CKG.git

Once this is finished, you can find all the Python modules necessary to run the Clinical Knowledge Graph in ``requirements.txt``.
To install all the packages required, simply run:

.. code-block:: bash
	
	$ cd CKG/
	$ pip install -r requirements.txt

.. warning:: Make sure the virtual environment previously created is active before installing ``requirements.txt``.

Now that all the packages are correctly installed, you will have to create the appropriate directory architecture within the local copy of the cloned repository:

.. code-block:: bash

	$ python setup_CKG.py
	$ python setup_config_files.py

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where all errors and warnings related to the code will be written to.


Add CKG to *.bashrc*
---------------------

In order run the the Clinical Knowledge Graph, add the path to the code to your ``.bashrc`` (or ``.bash_profile``):

1. Open the .bashrc file.

#. Add the following lines to the file and save it:

.. code-block:: bash
	
	PYTHONPATH="${PYTHONPATH}:/path/to/folder/CKG/src/"
	export PYTHONPATH

Notice that the path should always finish with "/CKG/src/".


#. To reload the bash file, first deactivate the virtual environment, reload ~/.bashrc, and activate the virtual environment again:

.. code-block:: bash
	
	$ deactivate
	$ source ~/.bashrc
	$ source path/to/env_name/bin/activate



.. figure:: ../_static/images/snomed_folder.png
    :width: 240px
    :align: right

    SNOMED-CT ontology folder.

.. _Build Neo4j graph database:

Build Neo4j graph database
---------------------------

The building of the CKG database is thoroughly automated. Most of the biomedical databases and ontology files will automatically be downloaded during building of the database. However, the following licensed databases have to be downloaded manually.


.. figure:: ../_static/images/drugbank_folder.png
    :width: 240px
    :align: right

    DrugBank database folder.


- `PhosphoSitePlus <https://www.phosphosite.org/staticDownloads>`__: *Acetylation_site_dataset.gz*, *Disease-associated_sites.gz*, *Kinase_Substrate_Dataset.gz*, *Methylation_site_dataset.gz*, *O-GalNAc_site_dataset.gz*, *O-GlcNAc_site_dataset.gz*, *Phosphorylation_site_dataset.gz*, *Regulatory_sites.gz*, *Sumoylation_site_dataset.gz* and *Ubiquitination_site_dataset.gz*.

- `DrugBank <https://www.drugbank.ca/releases/latest>`__: *All drugs* (under *COMPLETE DATABASE*) and *DrugBank Vocabulary* (under *OPEN DATA*).


.. figure:: ../_static/images/psp_folder.png
    :width: 240px
    :align: right

    PhosphoSitePlus database folder.

- `SNOMED-CT <https://www.nlm.nih.gov/healthit/snomedct/international.html>`__: *Download RF2 Files Now!*.

After download, move the files to their respective folders:

- PhosphoSitePlus: ``CKG/data/databases/PhosphoSitePlus``
- DrugBank: ``CKG/data/databases/DrugBank``
- SNOMED-CT: ``CKG/data/ontologies/SNOMED-CT``


In the case of SNOMED-CT, unzip the downloaded file and copy all the subfolders and files to the ``SNOMED-CT`` folder.


.. warning:: These three databases require login and authentication. To sign up go to `PSP Sign up <https://www.phosphosite.org/signUpAction>`__, `DrugBank Sign up <https://www.drugbank.ca/public_users/sign_up>`__ and `SNOMED-CT Sign up <https://uts.nlm.nih.gov/license.html>`__. In the case of SNOMED-CT, the UMLS license can take several business days.

.. note:: If the respective database folder is not created, please do it manually.

The last step is to build the database, which can be done using the ``builder.py`` module or a ``dump file``.


From builder.py
^^^^^^^^^^^^^^^^^^

To build the graph database, run ``builder.py``:

.. code-block:: bash
	
	$ cd src/graphdb_builder/builder
	$ python builder.py -b full -u neo4j

.. warning:: Before running ``builder.py``, please make sure your Neo4j graph is running. The builder will fail otherwise.

This action will take aproximately 6 hours but depending on a multitude of factors, it can take up to 10 hours.


From the provided dump file **(Testing)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dump file of the database is also made available in this `link <https://data.mendeley.com/datasets/mrcf7f4tc2/1>`__ and alternatively, you can use it to load the graph database contained in it. To do so, download both files (``ckg_080520.dump`` and ``data.tar.gz``). 

The ``.dump`` file will be used to load the Neo4j graph database:

1. Create ``backups`` and ``graph.db`` folders:

.. code-block:: bash

	$ cd /path/to/neo4jDatabases/database-identifier/installation-x.x.x/
	$ mkdir backups
	$ mkdir backups/graph.db
	$ cp 2019-11-04.dump backups/graph.db/.


2. After copying the dump file to backups/graph.db/, make sure the graph database is shutdown and run:

.. code-block:: bash
	
	$ bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force


In some systems you might have to run this as root:

.. code-block:: bash
	
	$ sudo bin/neo4j-admin load --from=backups/graph.db/2019-11-04.dump --database=graph.db --force
	$ sudo chown -R username data/databases/graph.db/


.. warning:: Make sure the dump file naming in the command above, matches the one provided to you.


3. Once you are done, start the database and you will have a functional graph database.


Be aware the database contained in the dump file **does NOT** include the licensed databases (**PhosphoSitePlus**, **DrugBank** and **SNOMED-CT**).

.. figure:: ../_static/images/data_folder.png
    :width: 240px
    :align: right

    Final CKG/data folder architecture.


To add the missing ontology and databases, as well as their dependencies (relationships to other nodes), please manually download the files as explained in :ref:`Build Neo4j graph database`, unzip the downloaded file ``data.tar.gz`` and place its contents in ``CKG/data/``. The folder ``data`` should look like the figure depicted.

Once this is done, run the following commands:

.. code-block:: bash
	
	$ cd CKG/src/graphdb_builder/builder
	$ python builder.py -b minimal -u username


.. note:: Remember of replace the ``username`` in each command, with your own neo4j username.


More on the dump file
^^^^^^^^^^^^^^^^^^^^^^^^^

Another great use for the dump file, is to generate backups of the database (e.g. different versions of the imported biomedical databases).
To generate a dump file of a specific Neo4j database, simply run:

.. code-block:: bash

	$ cd /path/to/neo4jDatabases/database-identifier/installation-x.x.x/
	$ bin/neo4j-admin dump --database=neo4j --to=backups/graph.db/name_of_the_file.dump


.. warning:: Remember to replace ``name_of_the_file`` with the name of the dump file you want to create.








