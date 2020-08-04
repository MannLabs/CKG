.. ClinicalKnowledgeGraph documentation master file, created by
   sphinx-quickstart on Wed Nov 27 16:10:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ClinicalKnowledgeGraph's documentation!
==================================================

This web page contains the documentation for the Python code using **Sphinx**.

.. toctree::
   :maxdepth: 2

   INTRO


First steps
------------

Are you new to the Clinical Knowledge Graph? Learn about how to use it and
all the possibilities.

* **Getting started**:
  :doc:`With Requirements <intro/getting-started-with-requirements>` |
  :doc:`With Neo4j <intro/getting-started-with-neo4j>` |
  :doc:`With Clinical Knowledge Graph <intro/getting-started-with-build>` |
  :doc:`With Windows <intro/getting-started-with-windows>` |
  :doc:`With Docker <intro/getting-started-with-docker>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: First steps

   intro/getting-started-with-requirements
   intro/getting-started-with-neo4j
   intro/getting-started-with-build
   intro/getting-started-with-windows
   intro/getting-started-with-docker


Getting started
-------------------------------------------------

* **Connecting to the CKG**:
  :doc:`Connect to DB <getting_started/connect-to-ckg>`

* **Create a new user in the graph database**:
  :doc:`Create new user <getting_started/create-new-user>`

* **Create a project in the database**:
  :doc:`Project Creation <getting_started/create-new-project>`

* **Upload experimental data**:
  :doc:`Data Upload <getting_started/upload-data>`

* **Define data analysis settings**:
  :doc:`Configuration <getting_started/data-analysis-config>`

* **Access the analysis report**:
  :doc:`Access report <getting_started/access-report>`

* **Report notification**:
  :doc:`Notifications <getting_started/notifications>`


.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Getting started

  getting_started/connect-to-ckg
  getting_started/create-new-user
  getting_started/create-new-project
  getting_started/upload-data
  getting_started/data-analysis-config
  getting_started/access-report
  getting_started/notifications


The project report
------------------

* **Generate a project**:
  :doc:`Project <project_report/project-report>`

* **The Tabs**:
  :doc:`Project tabs <project_report/project-tabs>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Project Report

   project_report/project-report
   project_report/project-tabs


CKG Builder
-----------

* **Ontology sources and parsers**:
  :doc:`Ontologies <ckg_builder/ontologies>`

* **Biomedical databases and resources**:
  :doc:`Databases <ckg_builder/databases>`

* **Parsing experimental data**:
  :doc:`Experiments <ckg_builder/experiments>`

* **Building the graph database from one module**:
  :doc:`Builder <ckg_builder/graphdb-builder>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: CKG Builder

   ckg_builder/ontologies
   ckg_builder/databases
   ckg_builder/experiments
   ckg_builder/graphdb-builder


Advanced features
------------------

* **CKG Statistics**:
  :doc:`Imports stats <advanced_features/import-statistics>` |
  :doc:`Graph database stats <advanced_features/graphdb-statistics>`

* **Jupyter notebooks**:
  :doc:`Notebooks <advanced_features/ckg-notebooks>`

* **Retrieving data from the CKG**:
  :doc:`DB Querying <advanced_features/ckg-queries>`

* **Data Analysis**:
  :doc:`Analysis <advanced_features/standard-analysis>`

* **Visualization**:
  :doc:`Plots <advanced_features/visualization-plots>`

* **R interface**:
  :doc:`R wrapper <advanced_features/R-interface>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Features

   advanced_features/import-statistics
   advanced_features/graphdb-statistics
   advanced_features/ckg-notebooks
   advanced_features/ckg-queries
   advanced_features/standard-analysis
   advanced_features/visualization-plots
   advanced_features/R-interface


System Requirements
---------------------

.. toctree::
   :maxdepth: 2

   system_requirements



API Reference
-------------

.. toctree::
   :maxdepth: 4

   autosummary/src


Project Info
-------------

.. toctree::
   :maxdepth: 2

   MANIFEST


Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
