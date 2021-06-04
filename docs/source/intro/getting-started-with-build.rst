.. _Installing CKG python library:

.. include:: ../global.rst

Installing CKG python library
===================================

Setting up the Clinical Knowledge Graph is straightforward.
Assuming you have **Python** |python_version| already installed and a virtual environment created following instructions here: :ref:`Installation`.



Setting up the Clinical Knowledge Graph
-----------------------------------------

The first step in setting up the CKG, is to obtain the complete code by cloning the GitHub repository:

.. code-block:: bash

	$ git clone https://github.com/MannLabs/CKG.git

Another option is to download it from the github page directly:

1. Go to https://github.com/MannLabs/CKG

2. In `Code` select **Download ZIP**

3. Unzip the file

Once this the cloning is finished or the file is unzipped, you can install CKG by running:

.. code-block:: bash

	$ cd CKG/
	$ conda activate ckgenv
	$ python setup.py install

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where all errors and warnings related to the code will be written to.
Further, it will create an executable file with CKG's app. To start the app, simpy run:

.. code-block:: bash

	$ ckg_app


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


From the provided dump file
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

	$ bin/neo4j-admin load --from=backups/graph.db/ckg_080520.dump --database=graph.db --force


In some systems you might have to run this as root:

.. code-block:: bash

	$ sudo bin/neo4j-admin load --from=backups/graph.db/ckg_080520.dump --database=graph.db --force
	$ sudo chown -R username data/databases/graph.db/


.. warning:: Make sure the dump file naming in the command above, matches the one provided to you.


3. Once you are done, start the database and you will have a functional graph database.


Be aware the database contained in the dump file **does NOT** include the licensed databases (**PhosphoSitePlus**, **DrugBank** and **SNOMED-CT**).

.. figure:: ../_static/images/data_folder.png
	:alt: data_folder
    :width: 240px
    :align: right

    Final CKG/data folder architecture.


To add the missing ontology and databases, as well as their dependencies (relationships to other nodes), please manually download the files as explained in :ref:`Build Neo4j graph database`, unzip the downloaded file ``data.tar.gz`` and place its contents in ``CKG/data/``. The folder ``data`` should look like the figure depicted.

Once this is done, run the following commands:

.. code-block:: bash

	$ cd CKG/ckg/graphdb_builder/builder
	$ python builder.py -b minimal -u username


.. note:: Remember of replace the ``username`` in each command, with your own neo4j username.



From builder.py
^^^^^^^^^^^^^^^^^^

To build the graph database, run ``builder.py``:

.. code-block:: bash

	$ cd ckg/graphdb_builder/builder
	$ python builder.py -b full -u neo4j

.. warning:: Before running ``builder.py``, please make sure your Neo4j graph is running. The builder will fail otherwise.

This action will take approximately 6 hours but depending on a multitude of factors, it can take up to 10 hours.


More on the dump file
^^^^^^^^^^^^^^^^^^^^^^^^^

Another great use for the dump file, is to generate backups of the database (e.g. different versions of the imported biomedical databases).
To generate a dump file of a specific Neo4j database, simply run:

.. code-block:: bash

	$ cd /path/to/neo4jDatabases/database-identifier/installation-x.x.x/
	$ bin/neo4j-admin dump --database=neo4j --to=backups/graph.db/name_of_the_file.dump


.. warning:: Remember to replace ``name_of_the_file`` with the name of the dump file you want to create.
