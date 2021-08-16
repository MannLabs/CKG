.. _notebooks:

The Clinical Knowledge Graph Notebooks
=======================================

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
the use of multiple visualisation tools. These notebooks can be found in ``ckg/notebooks``, under ``reporting`` or ``development``.


.. note:: If you would like to use two instances of the same notebook, just duplicate in-place and modify the name accordingly.

.. warning:: If the Clinical Knowledge Graph is deployed in a server, please set up a Jupyter Hub in order to allow access to the Jupyter Notebook.


Recipes notebooks
---------------------

In ``ckg/notebooks/recipes`` we gathered Jupyter notebooks with analysis and workflows we believe are of interest for the users but are still under development.

.. toctree::
	:glob:

	../notebooks/recipes/*.ipynb

- :doc:`**Access Project Report** <../notebooks/recipes/Access Project Report>`

Easy access to all the projects in the graph database. Loads all the data from a specific project (e.g. "P0000001") and shows the report in the notebook.
This notebook enables the visualisation of all the plots and tables that constitute a report, as well as all the dataframes used and produced during its generation.
By accessing the data directly, you can use the python functionality to further the analysis or visualisation.


- :doc:`**Working with R** <../notebooks/recipes/working_with_R>`

Notebook entirely written in R. One of the many advantages of using Jupyter notebooks is the possibility of writing in different programming languages.
In this notebook, we demonstrate how R can be used to, similarly to *project_reporting.ipynb*, load a project and explore the analysis and plots.

In the beginning of the notebook, we create custom functions to load a project, read the report, read a dataset, and plot and network from a json file.
Other R functions like these can be developed by the users according to their needs.


- :doc:`**Parallel plots** <../notebooks/recipes/Parallel plots>`

An example of a new interactive visualisation method, not currently implemented in the Clinical Knowledge Graph, but using data from a stored project. We start by loading\
all the data and the report of a specific project (e.g. "P0000001"), and accessing different dataframes within the proteomics dataset, as well as, the correlation network.\
This plot is then converted into a Pandas DataFrame and used as input for the interactive Parallel plot.

The function is created and made interactive with Jupyter Widgets ``interact`` function (``ipywidgets.interact``), which automatically creates user interface (UI) controls\
within the notebook. In this case, the user can select different clusters of proteins (from the correlation network) and observe their variation across groups.


- :doc:`**Download PRIDE data** <../notebooks/recipes/Download_PRIDE_data>`

Easily download data directly from PRIDE (https://www.ebi.ac.uk/pride/) by specifying the PRIDE identifier and the file name to download. The notebook also shows how to\
format the data and analyze them with CKG.

- :doc:`**Power Analysis** <../notebooks/recipes/Power Analysis>`

Power anlysis based on an existing project in CKG. It allows to define the sample size needed to achieve a specific statistical power using effect sizes from\
previous analyses.


- :doc:`**single sample Gene Set Enrichment Analysis** <../notebooks/recipes/ssGSEA_with_PCA>`

Perform single sample Gene Set Enrichment Analysis (ssGSEA) using different gene/protein sets and visualize them using Principal Component Analysis (PCA).

- :doc:`**Annotate Proteins with CKG Knowledge** <../notebooks/recipes/Knowledge from list of proteins>`

This notebook shows how to extract knowledge associated with a list of proteins and summarize it using different methods: centrality, pagerank.

- :doc:`**Annotate drugs with CKG Knowledge** <../notebooks/recipes/Knowledge from list of drugs>`

This notebook shows how to extract knowledge associated with a list of drugs and summarize it using different methods: centrality, pagerank.

- :doc:`**Batch effect correction** <../notebooks/recipes/Batch_correction>`

This notebook exemplifies how to correct batch effects in a proteomics experiment and how to add this correction in CKG's analytical pipeline (configuration).

- :doc:`**Convert SDRF format to CKG** <../notebooks/recipes/Converting_sdrf_to_CKG>`

This notebook shows CKG's functionality to convert SDRF format into CKG and viceversa.

- :doc:`**Upload a SDRF file to CKG** <../notebooks/recipes/Uploading_sdrf_to_CKG>`

How to upload programmatically a SDRF file into CKG.

- :doc:`**Convert MzTab format to CKG** <../notebooks/recipes/Read mztab>`

This notebook shows CKG's functionality to read MzTab format and convert it into CKG proteomics/metabolomics format (edge lists).

	

Reports notebooks
-------------------

Reports notebooks 


.. toctree::
	:maxdepth:1

   ../../notebooks/reports/*.ipynb



- :doc:`**Plasma proteome profiling discovers novel proteins associated with non‐alcoholic fatty liver disease** <../notebooks/reports/Plasma proteome profiling discovers novel proteins associated with non‐alcoholic fatty liver disease>`

A Jupyter notebook reanalyzing the study Plasma proteome profiling discovers novel proteins associated with non‐alcoholic fatty liver disease (https://www.embopress.org/doi/full/10.15252/msb.20188793).
The analyses shown in the notebook reproduce CKG's default analytical pipeline from data processing to knowledge.


- :doc:`**Urachal Carcinoma Case Study** <../notebooks/reports/Urachal Carcinoma Case Study>`


Jupyter notebook depicting the use of the Clinical Knowledge Graph database and the analytics core as a decision support tool, proposing a drug candidate in a specific subject case.

The project is analysed with the standard analytics workflow and a list of significantly differentially expressed proteins is returned. To further this analysis, we first filter\
for regulated proteins that have been associated to the disease in study (lung cancer); we then search the database for known inhibitory drugs for these proteins; and to narrow down\
the list, we can query the database for each drug's side effects. The treatment regimens are also available in the database and their side effects can be used to rank the proposed drugs.\
We can prioritise drugs with side effects dissimilar to the ones that caused an adverse reaction in the patient, and identify papers where these drugs have already been associated to the\
studied disease, further reducing the list of potential drugs candidates.


- :doc:`**Proteomics-Based Comparative Mapping of the Secretomes of Human Brown and White Adipocytes** <../notebooks/reports/Proteomics-Based Comparative Mapping of the Secretomes of Human Brown and White Adipocytes>`

A Jupyter notebook that exemplifies how to analyse any external dataset deposited in EBI's PRIDE database, by simply using the respective PRIDE identifier for the project (PXD...)\
and CKG's Analytics Core.

The entire project folder is downloaded from PRIDE, decompressed, and the proteinGroups.txt file is parsed to obtain relevant information including LFQ intensities. After being converted into\
a wide-format dataframe, the default analysis pipeline implemented in the CKG, is reproduced by calling the respective functions directly from the Analytics Core modules\
(*analytics_core.analytics* and *analytics_core.viz*).

Additionally, we demonstrate how other publicly accessible biomedical databases can be downloaded into the notebook, and their information mined for relevant metadata. In this specific case,
the Human Protein Atlas is downloaded and mined in order to filter for known or predicted secreted proteins.


- :doc:`**Covid-19 Olink Analysis from Massachusetts General Hospital** <../notebooks/reports/Olink analysis>`

A Jupyter notebook showing how CKG can be used to analyze a dataset generated with a different technology, in this case Olink.

The analysis reproduces similar results to the ones described in this manuscript: https://www.biorxiv.org/content/10.1101/2020.11.02.365536v2
and also extends these results by comparing different severity groups in this cohort based on WHO scores.


- :doc:`**Meduloblastoma Data Proteomics Re-analysis** <../notebooks/reports/Meduloblastoma Data Proteomics Re-analysis>`
- :doc:`**Meduloblastoma Data Analysis-SNF** <../notebooks/reports/Meduloblastoma Data Analysis-SNF>`

These two Jupyter notebooks show:

1) A re-analysis of this study: Proteomics, Post-translational Modifications, and Integrative Analyses Reveal Molecular Heterogeneity within Medulloblastoma Subgroups (https://www.sciencedirect.com/science/article/pii/S1535610818303581).
   This analysis uses the proteomics data to compare the clinical Subgroups
2) A multiomics approach integrating all the molecular data available: proteomics, RNA sequencing and PTMs. CKG uses Similarity Network Fusion (https://www.nature.com/articles/nmeth.2810) to integrate these datasets and to define Medulloblastoma subgroups. 
   The results are further explained in this `presentation_meduloblastoma`_.

.. _presentation_meduloblastoma: https://github.com/MannLabs/CKG/blob/master/ckg/notebooks/reports/Archer_et_al_2018/Medulloblastoma_presentation.pdf



- :doc:`**CPTAC Glioblastoma Discovery Study Proteomics Re-analysis** <../notebooks/reports/CPTAC_Glioblastoma_Discovery_Study>`

A re-analysis of the CPTAC Discovery Study using the Proteomics data of 100 brain tumor samples and 10 normal samples (https://cptac-data-portal.georgetown.edu/study-summary/S048). More details on the results available in this `presentation_gbm`_.

.. _presentation_gbm: https://github.com/MannLabs/CKG/blob/master/ckg/notebooks/reports/CPTAC_GBM_discovery_strudy/GBM_results.pdf

