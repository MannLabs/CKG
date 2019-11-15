Getting Started with Neo4j 
==========================

Getting started with Neo4j is easy.

First download the desktop version from https://neo4j.com/download/.
The community version of the software is free but a sign up is required.


Create a new Local Graph.
Install APOC and GRAPH ALGORITHMS.
Modify Settings: comment the option 'dbms.directories.import=import'.
Start the Graph.


Add Neo4j graph database to *.bashrc*
-------------------------------------

In order run the graph database, add the path to the database to your .bashrc (or .bash_profile):

1. Open the .bashrc file.

2. Depending on your system, the path may vary. To check the path to the database go to 'Logs' in the Neo4j Desktop interface.

2. Add the following lines to the file and save it:

NEO4J_HOME="/Users/username/Library/Application Support/Neo4j Desktop/Application/neo4jDatabases/database-identifier/installation-3.5.6/"
export NEO4J_HOME


3. Reload .bashrc:

.. prompt:: bash $

	source ~/.bashrc








