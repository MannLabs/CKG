Using Jupyter Notebooks with the Clinical Knowledge Graph
=========================================================

The Jupyter Notebook is used to interact with the notebooks provided in the Clinical Knowledge Graph.
This open-source application allows you to create and share code, visualise outputs and integrated multiple big data tools.

In order to get started, make sure you have Python installed (3.3 or greater) as well as Jupyter Notebook. The latter can be installed using pip (see below).\
For more detailed instructions, visit the `official guide <https://jupyter.org/install.html>`_.

.. code-block:: bash

	$ python3 -m pip install --upgrade pip
	$ python3 -m pip install jupyter

Congratulations! Now you can run the notebook, by typing the following command in the Terminal (Mac/Linux):

.. code-block:: bash

	$ jupyter notebook

Or,

.. code-block:: bash

	$ jupyter-notebook


As part of the Clinical Knowledge Graph package, we provide a series of Jupyter notebooks to facilitate the analysis of data, database querying and \
the use of multiple visualisation tools. These notebooks can be found in ``src/notebooks``, under ``reporting`` or ``development``.


.. note:: If you would like to use two instances of the same notebook, just duplicate in-place and modify the name accordingly.

.. warning:: If the Clinical Knowledge Graph is deployed in a server, please set up a Jupyter Hub in order to allow access to the Jupyter Notebook.


Reporting notebooks
-------------------

Reporting notebooks refers to Jupyter notebooks that have been finished and properly tested by the developers, and are ready to be used by the community.

- **project_reporting.ipynb**

Easy access to all the projects in the graph database. Loads all the data from a specific project (e.g. "P0000001") and shows the report in the notebook.
This notebook enables the visualisation of all the plots and tables that constitute a report, as well as all the dataframes used and produced during its generation.
By accessing the data directly, you can use the python functionality to further the analysis or visualisation.

- **working_with_R.ipynb**

Notebook entirely written in R. One of the many advantages of using Jupyter notebooks is the possibility of writing in different programming languages.
In this notebook, we demonstrate how R can be used to, similarly to *project_reporting.ipynb*, load a project and explore the analysis and plots.

In the beginning of the notebook, we create custom functions to load a project, read the report, read a dataset, and plot and network from a json file.
Other R functions like these can be developed by the users according to their needs.


- **Parallel plots.ipynb**

An example of a new interactive visualisation method, not currently implemented in the Clinical Knowledge Graph, but using data from a stored project. We start by loading all the data and the report of a specific project (e.g. "P0000001"), and accessing different dataframes within the proteomics dataset, as well as, the correlation network. This plot is then converted into a Pandas DataFrame and used as input for the interactive Parallel plot.

The function is created and made interactive with Jupyter Widgets ``interact`` function (``ipywidgets.interact``), which automatically creates user interface (UI) controls within the notebook. In this case, the user can select different clusters of proteins (from the correlation network) and observe their variation across groups.

- **Urachal Carcinoma Case Study.ipynb**

Jupyter notebook depicting the use of the Clinical Knowledge Graph database and the analytics core as a decision support tool, proposing a drug candidate in a specific subject case.

The project is analysed with the standard analytics workflow and a list of significantly differentially expressed proteins is returned. To further this analysis we first filter for regulated proteins that have been associated to the disease in study (lung cancer); we then search the database for known inhibitory drugs for these proteins; and to narrow down the list, we can query the database for each drug's side effects. The treatment regimens are also available in the database and their side effects can be used to rank the proposed drugs. We can prioritise drugs with side effects dissimilar to the ones that caused an adverse reaction in the patient, and identify papers where these drugs have already been associated to the studied disease, further reducing the list of potential drugs candidates.


Development notebooks
---------------------

In ``src/report_manager/development`` we gathered Jupyter notebooks with analysis and workflows we believe are of interest for the users but are still under development.

When a notebook in this folder is functional and successfully benchmarked, the notebook is moved to the reporting directory.
