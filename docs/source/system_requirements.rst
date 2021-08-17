System Requirements
=====================

The Clinical Knowledge Graph was conceived as a multi-user platform and therefore requires installation in a server-like setup and data systems administration knowledge. However, individual users can have local instances of the CKG, making sure data, software and hardware requirements are fulfilled.


Data
------------------

Licensed databases used by the CKG package require login and authentication in order to download their data. This is the case of **SNOMED-CT**, **DrugBank** and **PhosphoSitePlus**.
Make sure you sign up to these three databases well in advance as the licensing process can take several days to conclude.

To sign up go to `PSP Sign up <https://www.phosphosite.org/signUpAction>`__, `DrugBank Sign up <https://www.drugbank.ca/public_users/sign_up>`__ and `SNOMED-CT Sign up <https://uts.nlm.nih.gov/license.html>`__, and follow the instructions.

Once you have been given authorization to access the data, please download the files as follows:

- `PhosphoSitePlus <https://www.phosphosite.org/staticDownloads>`__: *Acetylation_site_dataset.gz*, *Disease-associated_sites.gz*, *Kinase_Substrate_Dataset.gz*, *Methylation_site_dataset.gz*, *O-GalNAc_site_dataset.gz*, *O-GlcNAc_site_dataset.gz*, *Phosphorylation_site_dataset.gz*, *Regulatory_sites.gz*, *Sumoylation_site_dataset.gz* and *Ubiquitination_site_dataset.gz*.

- `DrugBank <https://www.drugbank.ca/releases/latest>`__: *All drugs* (under *COMPLETE DATABASE*) and *DrugBank Vocabulary* (under *OPEN DATA*).

- `SNOMED-CT <https://www.nlm.nih.gov/healthit/snomedct/international.html>`__: *Download RF2 Files Now!*.

These files will be later used in :ref:`Build Neo4j graph database`.


Software
-------------------

- Python 3.7.9
- Redis server
- Neo4j Desktop
- Neo4j database == 4.2.3


Hardware
--------------------

- Memory: 16Gb
- Disk space: >= 200Gb
- Stable internet connection


.. note:: When building the Docker image further disk space is needed. We recomend allocating 300Gb, although the final CKG image will be around 150Gb.