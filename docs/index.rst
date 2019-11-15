.. ClinicalKnowledgeGraph documentation master file, created by
   sphinx-quickstart on Wed Oct  9 23:03:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ClinicalKnowledgeGraph's documentation!
==================================================

This web page contains the documentation for the Python code using **Sphinx**.


First steps
------------

Are you new to the Clinical Knowledge Graph? Learn about how to use it and
all the possibilities.

* **Getting started**:
  :doc: 'With Neo4j <intro/getting-started-with-neo4j>' |
  :doc: 'With Clinical Knowledge Graph <intro/getting-started-with-build>'
.. builder.py + setup_*.py + log (where errors will be recorded)
.. Alternative: Docker



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: First steps

   intro/getting-started-with-neo4j
   intro/getting-started-with-build



Getting started with the Clinical Knowledge Graph
-------------------------------------------------

* **Connecting to the CKG**:
  :doc:
.. graphdb_connector

* **Create a new user in the graph database**:
  :doc: 'Create new user <>'
.. create_user.py

* **Create a project in the database**:
  :doc: 'Project Creation <>'
.. project creation stuff + queue

* **Upload experimental data**:
  :doc: 'Data Upload <>'

* **Define data analysis settings**:
  :doc: 'Clinical data <>' |
  :doc: 'Proteomics <>' |
  :doc: 'Whole exome sequencing <>' |
  :doc: 'Multiomics <>'

* **Access the analysis report**:
  :doc: 'Dash web app <>' |
  :doc: 'Jupyter notebook <>'
.. basicApp.py + intialApp.py + apps_config.py + app.py + index.py

* **Report notification**:
  :doc:
.. utils.py (slack notification)



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
  :doc:
.. project.py + project_config.py + report.py + projectApp

* **The Tabs**:
  :doc:
.. dataset.py + knowledge.py



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Project Report

   project_report/project-report
   project_report/project-tabs



CKG Builder
-----------

* **Ontology sources and parsers**:
  :doc:
.. ontoogies *

* **Biomedical databases and resources**:
  :doc:
.. databases *

* **Parsing experimental data**:
  :doc:
.. experiments *

* **Building the graph database from one module**:
  :doc:
.. builder + buider_utils + mapping + importer + loader



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: CKG Builder

   ckg_builder/ontologies
   ckg_builder/databases
   ckg_builder/experiments
   ckg_builder/graphdb-builder



Advanced featues
----------------

* **CKG Statistics**:
  :doc: 'Imports stats <>' |
  :doc: 'Graph database stats <>'
.. honepage* + imports*

* **Jupyter notebooks**:
  :doc: 'Reporting notebooks <>' |
  :doc: 'Development notebooks <>'
.. all the notebooks

* **Retrieving data from the CKG**:
  :doc:
.. queries folder + cypher.yml

* **Data Analysis**:
  :doc:
.. analyses + analysisResult

* **Visualization**:
  :doc:
.. plots

* **R interface**:
  :doc:
.. notebook + R_packages.R + R2Py.py



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Features

   advanced_features/import-statistics
   advanced_features/graphdb-statistics
   advanced_features/reporting-notebooks
   advanced_features/development-notebooks
   advanced_features/ckg-queries
   advanced_features/standard-analysis
   advanced_features/visualization-plots
   advanced_features/R-interface



System requirements
-------------------
.. requirements.txt

* **Mac OS X**:
  :doc:

* **Linux**:
  :doc:

* **Windows**:
  :doc:
.. docker



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: System Requirements

   system_require/mac-os
   system_require/linux
   system_require/windows



.. _apiref:

API Reference
-------------

.. toctree::
   :maxdepth: 3

   src



Project Info
------------

.. toctree::
   :maxdepth: 2

   CONTRIBUTING
   AUTHORS
   HISTORY
   BACKERS
   CODE_OF_CONDUCT



Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
