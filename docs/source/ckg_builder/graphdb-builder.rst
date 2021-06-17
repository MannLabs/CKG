

##############################
Building CKG's Graph Database
##############################

CKG has a dedicated module (graphdb_builder) that can be used to generate the entire Knowledge Graph (full update) or to update specific ontologies, databases or experiments (partial update).
The module works as a 2-step process:


1) Import:
   uses specific parsers and configuration files to convert `ontologies <https://github.com/MannLabs/CKG/tree/master/ckg/graphdb_builder/ontologies>`__, `databases <https://github.com/MannLabs/CKG/tree/master/ckg/graphdb_builder/databases>`__ and `experimental data (projects) <https://github.com/MannLabs/CKG/tree/master/ckg/graphdb_builder/experiments>`__ into tab-separated values files with the entities (nodes) and relationships to be imported into the graph database. The generated files are stored in the `data/imports` directory.


2) Loading: 
   When loading data into the graph, we need to specify the type of node (entity) to be loaded. The types of nodes correspond to the defined data model:
   
    .. image:: ../_static/images/data_model.png
        :width: 100%
        :align: center

    
   The list of entities to be loaded into the graph is defined in the `builder configuration file <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_builder/builder/builder_config.yml>`__ under graph.  
   
   CKG has predefined `Cypher queries <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_builder/builder/cypher.yml>`__ to load the generated tsv files into the graph database. To facilitate the use and understanding of the queries, they are defined in YAML format, which allows to define attributes such as query name, description, nodes and relationships involved.



.. warning:: Remember that the graph database needs to be running when the database is being built or updated (Loading step).


Graph Database Builder
=======================
.. image:: ../_static/images/graphdb_builder.png
    :width: 100%
    :align: center


.. _full:

Full update
^^^^^^^^^^^^


The full update goes through both steps and updates ontologies, databases and the available experiments. There are several options to run the full update:

**1) Command-line (executable)** *(Recommended)*

.. code-block:: bash

    $ ckg_build

This will initiate the full update with default parameters:

- `download=True` -- ontologies and databases will be downloaded from their sources

- `n_jobs=3` -- 3 processes will be use simultaneously

**2) Command-line (programmatically)**
   
The module `graphdb_builder/builder/builder.py <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_builder/builder/builder.py>`__ can be called as a python script in the command-line. Use `-h` to get help on how to use it:

.. code-block:: bash

    $ python builder.py -h
    
        usage: builder.py [-h] [-b {import,load,full,minimal}]
                [-i {experiments,databases,ontologies,users} [{experiments,databases,ontologies,users} ...]]
                [-l LOAD_ENTITIES [LOAD_ENTITIES ...]] [-d DATA [DATA ...]]
                [-s SPECIFIC [SPECIFIC ...]] [-n N_JOBS] [-w DOWNLOAD] -u
                USER

        optional arguments:
        -h, --help            show this help message and exit
        -b {import,load,full,minimal}, --build_type {import,load,full,minimal}
                                define the type of build you want (import, load, full
                                or minimal (after dump file))
        -i {experiments,databases,ontologies,users} [{experiments,databases,ontologies,users} ...], --import_types {experiments,databases,ontologies,users} [{experiments,databases,ontologies,users} ...]
                                If only import, define which data types (ontologies,
                                experiments, databases, users) you want to import
                                (partial import)
        -l LOAD_ENTITIES [LOAD_ENTITIES ...], --load_entities LOAD_ENTITIES [LOAD_ENTITIES ...]
                                If only load, define which entities you want to load
                                into the database (partial load)
        -d DATA [DATA ...], --data DATA [DATA ...]
                                If only import, define which ontology/ies,
                                experiment/s or database/s you want to import
        -s SPECIFIC [SPECIFIC ...], --specific SPECIFIC [SPECIFIC ...]
                                If only loading, define which ontology/ies, projects
                                you want to load
        -n N_JOBS, --n_jobs N_JOBS
                                define number of cores used when importing data
        -w DOWNLOAD, --download DOWNLOAD
                                define whether or not to download imported data
        -u USER, --user USER  Specify a user name to keep track of who is building
                                the database



For a full update:

.. code-block:: bash

    $ python builder.py --build_type full --download True --user ckg_user

**3) Admin page (CKG app)**

In CKG's app, the `Admin page` provides as well the option to start a full update of the database.

.. image:: ../_static/images/admin_update.png
    :width: 100%
    :align: center

.. warning:: This option takes longer than the command-line options because multi-processing is not possible.

.. _partial:

Partial update
^^^^^^^^^^^^^^^


If you want to update a specific ontology, database or project, you can use the partial update functionality in graphdb_builder. The partial update is done programmatically only and you will need to define which ontologies or databases names, or project identifiers to update.
The ontologies and databases that are available to import can be checked in the configuration files: `ontologies_config.yml <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_builder/ontologies/ontologies_config.yml>`__ under `ontologies` and `databases_config.yml <https://github.com/MannLabs/CKG/blob/master/ckg/graphdb_builder/databases/databases_config.yml>`__ under `databases`.
The partial update is again a 2-step process, so you will need to import the ontologies/databases/projects and then load them into the graph. 

For instance, to update the Disease Ontology:

    .. code-block:: bash

        $ python builder.py --build_type import --import_types ontologies --data disease --download True --user ckg_user
        $ python builder.py --build_type load --load_entities ontologies --specific disease --user ckg_user

To update a database:

    .. code-block:: bash

        $ python builder.py --build_type import --import_types databases --data HMDB --download True --user ckg_user
        $ python builder.py --build_type load --load_entities metabolite --user ckg_user

