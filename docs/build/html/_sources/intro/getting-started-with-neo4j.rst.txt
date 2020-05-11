.. _Getting Started with Neo4j:

Getting Started with Neo4j
============================

Getting started with Neo4j is easy.

First download a copy of the Neo4j desktop version from the `Neo4j download page <https://neo4j.com/download/>`__.
The Community Edition of the software is free but a sign up is required.
Once the file has downloaded, you can install Neo4j by following the instructions automatically opened in the browser.

.. image:: ../_static/images/neo4j_app1.png
    :width: 32%
.. image:: ../_static/images/neo4j_app2.png
    :width: 32%
.. image:: ../_static/images/neo4j_app3.png
    :width: 32%

Open the Neo4j Desktop App and create a database by clicking :guilabel:`Add graph`, followed by :guilabel:`Create a Local Graph`, using the password "neo4j".
Now that your database is created:

.. image:: ../_static/images/neo4j_app4.png
    :width: 32%
.. image:: ../_static/images/neo4j_app5.png
    :width: 32%
.. image:: ../_static/images/neo4j_app6.png
    :width: 32%

1. Click :guilabel:`Manage` and then :guilabel:`Plugins`. Install "**APOC**" and "**GRAPH ALGORITHMS**".
#. Click the tab :guilabel:`Settings`, and comment the option ``dbms.directories.import=import`` by adding ``#`` at the beginning of the line.
#. Click :guilabel:`Apply` at the bottom of the window.
#. Start the Graph by clicking the play sign, at the top of the window.

If the database starts and no errors are reported in the tab :guilabel:`Logs`, you are redy go to!


Add Neo4j graph database to *.bashrc*
----------------------------------------

In order run the graph database, add the path to the database to your ``.bashrc`` (or ``.bash_profile``) file.

To find out which of the files your machine uses, go to the terminal and type ``more ~/.bash`` and double press the tab key on your keyboard for auto-complete. 
Immediately below, multiple filenames will be printed, check if among those, is ``.bashrc`` or ``.bash_profile``.

.. note:: The bash file can be name ``.bashrc`` or ``.bash_profile``. if your system does not have either, created one of them (e.g. vi ~/.bash_profile).

1. Open the ``.bash_profile`` (or ``.bashrc``) with your favourite text editor. In this case, we use the **vi** editor:

.. code-block:: bash
	
	$ vi ~/.bash_profile

.. note:: To edit with **vi** press ``i`` on your keyboard.

#. Add the path to the previously created Neo4j database to the file:

.. code-block:: bash

	NEO4J_HOME="/Users/username/Library/Application Support/Neo4j Desktop/Application/neo4jDatabases/database-identifier/installation-3.X.X/"
	export NEO4J_HOME

.. note:: To save and close a file with **vi** editor, press ``Esc`` followed by ``:wq``.

.. warning:: Depending on your system, the path may vary. To check the path to the database go to ``Logs`` in the Neo4j Desktop interface.

#. Reload the ``.bashrc`` (or ``.bash_profile``)  file:

.. code-block:: bash

	$ source ~/.bashrc

or

.. code-block:: bash
	
	$ source ~/.bash_profile












