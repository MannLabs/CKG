.. _Installing Neo4j:

.. include:: ../global.rst

Installing Neo4j
===================

Getting started with Neo4j is easy, just install Neo4j desktop and you are ready.

Neo4j Desktop
^^^^^^^^^^^^^^

You can download Neo4j Desktop from the `Neo4j download page <https://neo4j.com/download/>`__. Neo4j Desktop is bundled with a Java so you won't need to install Java separately. If you decide to install Neo4j directly, please check the requirements `here <https://neo4j.com/docs/operations-manual/current/installation/requirements/>`__.
The Community Edition of the software is free but a sign up is required.
Once the file has downloaded, you can install Neo4j by following the instructions automatically opened in the browser.

.. image:: ../_static/images/neo4j_app1.png
    :width: 32%
.. image:: ../_static/images/neo4j_app2.png
    :width: 32%
.. image:: ../_static/images/neo4j_app3.png
    :width: 32%

Open the Neo4j Desktop App and create a database by clicking :guilabel:`Add`, followed by :guilabel:`Local DBMS`, **choose database version**  |neo4j_version| using the password "NeO4J".
Now that your database is created:

.. image:: ../_static/images/neo4j_app5.png
    :width: 32%
.. image:: ../_static/images/neo4j_app4.png
    :width: 32%
.. image:: ../_static/images/neo4j_app6.png
    :width: 32%

1. Click :guilabel:`Manage` and then :guilabel:`Plugins`. Install "**APOC**" and "**Graph Data Science Library**".
#. Click the tab :guilabel:`Settings`, and comment the option ``dbms.directories.import=import`` by adding ``#`` at the beginning of the line.
#. Click :guilabel:`Apply` at the bottom of the window.
#. Start the Graph by clicking the play sign, at the top of the window.

.. image:: ../_static/images/neo4j_app7.png
    :width: 50%

If the database starts and no errors are reported in the tab :guilabel:`Logs`, you are ready go to!

