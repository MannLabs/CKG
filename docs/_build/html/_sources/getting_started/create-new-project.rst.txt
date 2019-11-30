Create a new project in the database
====================================

Preparations
------------

**Neo4j**

1. Open neo4j desktop
2. Start the database

**Terminal**

1. In one terminal window:

- Start a redis-server:

.. prompt:: bash $

  redis-server

2. In another terminal window:

- Navigate to CKG/src/report_manager/
- Start a celery queue from the report_manager directory:

.. prompt:: bash $

	celery -A worker worker -l info

3. In another terminal window:

- run the index:

.. prompt:: bash $

	python index.py

This will print some warnings, which should be okay

**Browser**

1. Go to ``http://localhost:5000/``.

2. Enter username and password


Project creation
----------------
**Browser**

Go to ``http://localhost:5000/apps/projectCreation``

This will direct you to the project creation page.
Fill in your project information.
Remember to press "Add" after selecting items from the drop-down menu (the page will update every time).
Press "Create Project" (you will now be able to find your project in the neo4j database)
Download the "Clinical Data template" and proceed to the 'Upload project experimental data'_.


Other
-----

**Neo4j**

> My Project > Manage > Open Browser > Project [Node Labels]
You can expand your project information if you click on your project node and next press the < in the bottom of the window
Here you will find your "id", typically something like "P00000**"

**Browser**

Go to:
  http://localhost:5000/apps/project/{YOUR_PROJECT_ID}

or go to:
  http://localhost:5000
  and select your project from the drop-down menu
