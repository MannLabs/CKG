Define data analysis parameters
===============================

A multitude of different analysis methods and visualisation plots have been implemented within the ``analytics_core`` of the Clinical Knowledge Graph.

The default workflow makes use of the functions defined in this module and runs, for each data type, the analysis pipeline defined in a configuration file. These configuration files are defined in YAML format (https://yaml.org/spec/1.2/spec.html), which can 
be easily read in Python into a dictionary structure with sections and analyses. For each analysis we need to define the data that will be used (i.e original data), how the results will be visualized (i.e pca_plot) and what parameters need to be used (i.e components: 2).


.. image:: ../_static/images/analytics_configuration.png
   :width: 75%
   :align: center



In the CKG, we have default analysis defined for Clinical data, Proteomics, and Multiomics. All the analysis configuration files can be modified to fit your project or data. 

To check how each configuration file looks like and how to modify them, please follow the links below.

.. toctree::
   :caption: Data configuration files

   data_settings/clinical-data
   data_settings/proteomics
   data_settings/multiomics
