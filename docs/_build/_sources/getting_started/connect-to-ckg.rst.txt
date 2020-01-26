Connecting to the Clinical Knowledge Graph database
===================================================

In order to make use of the CKG database you just built, we need to connect to it and be able to query for data.
This connection is established via one of Neo4j's Python drivers ``Py2neo``, a library and comprehensive toolkit developed to enable working with Neo4j from within Python applications, and should already be installed in your virtual environment.

Another essential tool when working with Neo4j databases, is the `Cypher query language <https://neo4j.com/developer/cypher-query-language/>`_. We recommend becoming familiar with it, to understand the queries used in the different analysis.


Py2neo connector
------------------

Within the CKG package, the ``graph_connector`` module was created to connect the different parts of the Python code, to the Neo4j database and allow their interaction.

In this module, the *Graph* class from ``py2neo`` is used to represent the graph data storage space within the Neo4j database, and a YAML configuration file is parsed to retrieve the connection details. The configuration file ``connector_config.yml`` contains the database server host name, server port, user to authenticate as, and password to use for authentication.


Python prompt
	from graphdb_connector import connector
	driver = connector.getGraphDatabaseConnectionConfiguration()

Once the connection is established, we can start querying the database. For example:


Python prompt
	example_query = 'MATCH (p:Project)-[:HAS_ENROLLED]-(s:Subject) RETURN p.id as project_id, COUNT(s) as n_subjects'
	results = connector.getCursorData(driver=driver, query=example_query, parameters={})

This query searches the database for all the available projects and counts how many subjects have been enrolled in each one, returning a pandas DataFrame with "project_id" and "n_subjects" as columns.
