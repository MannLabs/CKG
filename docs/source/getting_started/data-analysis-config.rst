Define data analysis parameters
===============================

A multitude of different analysis methods and visualisation plots have been implemented within the ``analytics_core`` of the Clinical Knowledge Graph.
The default workflow makes use of these resources and runs, for each data type, the analysis pipeline defined in a configuration file. In the CKG, we have
default analysis defined for Clinical data, Proteomics, and Multiomics. All the analysis configuration files can be modified to fit your project or data.

To check how each configuration file looks like and how to modify them, please follow the links below.

.. toctree::
   :caption: Data configuration files

   data_settings/clinical-data
   data_settings/proteomics
   data_settings/multiomics
