Connecting to the Clinical Knowledge Graph database
===================================================

In order to make use of the CKG database you just built, we need to connect to it and be able to query for data.
This connection is established via one of Neo4j's Python drivers ``Py2neo``, a library and comprehensive toolkit developed to enable working with Neo4j from within Python applications, and should already be installed in your virtual environment.

Another essential tool when working with Neo4j databases, is the `Cypher query language <https://neo4j.com/developer/cypher-query-language/>`__. We recommend becoming familiar with it, to understand the queries used in the different analyses.


Py2neo connector
------------------

Within the CKG package, the ``graph_connector`` module was created to connect the different parts of the Python code, to the Neo4j database and allow their interaction.

In this module, the *Graph* class from ``py2neo`` is used to represent the graph data storage space within the Neo4j database, and a YAML configuration file is parsed to retrieve the connection details. The configuration file ``connector_config.yml`` contains the database server host name, server port, user to authenticate as, and password to use for authentication.

.. code-block:: python

	from graphdb_connector import connector
	driver = connector.getGraphDatabaseConnectionConfiguration()

Once the connection is established, we can start querying the database. For example:

.. code-block:: python

	example_query = 'MATCH (p:Project)-[:HAS_ENROLLED]-(s:Subject) RETURN p.id as project_id, COUNT(s) as n_subjects'
	results = connector.getCursorData(driver=driver, query=example_query, parameters={})

This query searches the database for all the available projects and counts how many subjects have been enrolled in each one, returning a pandas DataFrame with "project_id" and "n_subjects" as columns.


Changing/Updating database connection
---------------------------------------

The connection to the graph database requires credentials, which are stored in ``graphdb_connector/connector_config.yml``.
This files includes the following lines:

.. code-block:: python

	db_url: "0.0.0.0"
	#dbPort = 7688 #Production environment
	db_port: 7687 #Test environment
	db_user: "neo4j"
	db_password: "NeO4J"

The initial password to create a new Neo4j database is set to **NeO4J**. If you would like to use another password when creating the database, you can edit the mentioned file and replace **NeO4J** with any other password of your choosing.
Another option is to change the password directly in the database by accessing :guilabel:`Manage` in the Neo4j desktop window, select the tab :guilabel:`Administration` and then set the new password.
Ultimately, make sure that the password in ``graphdb_connector/connector_config.yml`` and in the Neo4j database are the same.
