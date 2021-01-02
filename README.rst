.. image:: ./src/report_manager/assets/banner.jpg
    :align: center

.. image:: /_static/images/banner.jpg
    :align: center



**Clinical Knowledge Graph**  
============================
*version: 1.0*

A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.

* Documentation: `https://CKG.readthedocs.io <https://CKG.readthedocs.io>`_

* GitHub: `https://github.com/MannLabs/CKG <https://github.com/MannLabs/CKG>`_
* Free and open source software: `MIT license <https://github.com/MannLabs/CKG/LICENSE.rst>`_
* Reference: https://www.biorxiv.org/content/10.1101/2020.05.09.084897v1
* Graph Database dump file and additional relationships: https://data.mendeley.com/datasets/mrcf7f4tc2/1


Abstract
------------

.. image:: ./src/report_manager/assets/abstract.png
    :align: center

.. image:: /_static/images/abstract_figure.png
    :align: center

The promise of precision medicine is to deliver personalized treatment based on the unique physiology of each patient. This concept was fueled by the genomic revolution, but it is now evident that integrating other types of omics data, like proteomics, into the clinical decision-making process will be essential to accomplish precision medicine goals. However, quantity and diversity of biomedical data, and the spread of clinically relevant knowledge across myriad biomedical databases and publications makes this exceptionally difficult. To address this, we developed the Clinical Knowledge Graph (CKG), an open source platform currently comprised of more than 16 million nodes and 220 million relationships to represent relevant experimental data, public databases and the literature. The CKG also incorporates the latest statistical and machine learning algorithms, drastically accelerating analysis and interpretation of typical proteomics workflows. We use several biomarker studies to illustrate how the CKG may support, enrich and accelerate clinical decision-making.


Cloning and installing
-----------------------

.. 

	Installation requires >= 80 GB of disk space. See details `here <docs/source/system_requirements.rst>`_. 

The setting up of the CKG includes several steps and might take a few hours (if you are building the database from scratch). However, we have prepared documentation and manuals that will guide through every step.
To get a copy of the GitHub repository on your local machine, please open a terminal windown and run:

.. code-block:: bash

	$ git clone https://github.com/MannLabs/CKG.git

This will create a new folder named "CKG" on your current location. To access the documentation, use the ReadTheDocs link above, or open the html version stored in the *CKG* folder `CKG/docs/build/html/index.html`. After this, follow the instructions in "First Steps" and "Getting Started".

.. warning:: If git is not installed in your machine, please follow this `tutorial <https://www.atlassian.com/git/tutorials/install-git>`__ to install it.


Features
---------------

* Cross-platform: Mac, and Linux are officially supported. Instructions `for Windows <https://ckg.readthedocs.io/en/latest/intro/getting-started-with-windows.html>`_  exist.

* Docker container runs all neccessary steps to setup the CKG. 


Disclaimer 
---------------

This resource is intended for research purposes and must not substitute a doctor’s medical judgement or healthcare professional advice.


Important Note
---------------

The databases provided within the Clinical Knowledge Graph (CKG) have their own licenses and the use of CKG still requires compliance with these data use restrictions. Please, visit the data sources directly for more information:

+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Source type | Source                          | URL                                                        | Reference                                    |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | UniProt                         | https://www.uniprot.org/                                   | https://www.ncbi.nlm.nih.gov/pubmed/29425356 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | TISSUES                         | https://tissues.jensenlab.org/                             | https://www.ncbi.nlm.nih.gov/pubmed/29617745 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | STRING                          | https://string-db.org/                                     | https://www.ncbi.nlm.nih.gov/pubmed/30476243 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | STITCH                          | http://stitch.embl.de/                                     | https://www.ncbi.nlm.nih.gov/pubmed/26590256 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | SMPDB                           | https://smpdb.ca/                                          | https://www.ncbi.nlm.nih.gov/pubmed/24203708 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | SIGNOR                          | https://signor.uniroma2.it/                                | https://www.ncbi.nlm.nih.gov/pubmed/31665520 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | SIDER                           | http://sideeffects.embl.de/                                | https://www.ncbi.nlm.nih.gov/pubmed/26481350 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | RefSeq                          | https://www.ncbi.nlm.nih.gov/refseq/                       | https://www.ncbi.nlm.nih.gov/pubmed/26553804 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | Reactome                        | https://reactome.org/                                      | https://www.ncbi.nlm.nih.gov/pubmed/31691815 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | PhosphoSitePlus                 | https://www.phosphosite.org/                               | https://www.ncbi.nlm.nih.gov/pubmed/25514926 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | Pfam                            | https://pfam.xfam.org/                                     | https://www.ncbi.nlm.nih.gov/pubmed/30357350 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | OncoKB                          | https://www.oncokb.org/                                    | https://www.ncbi.nlm.nih.gov/pubmed/28890946 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | MutationDs                      | https://www.ebi.ac.uk/intact/resources/datasets#mutationDs | https://www.ncbi.nlm.nih.gov/pubmed/30602777 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | Intact                          | https://www.ebi.ac.uk/intact/                              | https://www.ncbi.nlm.nih.gov/pubmed/24234451 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | HPA                             | https://www.proteinatlas.org/                              | https://www.ncbi.nlm.nih.gov/pubmed/21572409 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | HMDB                            | https://hmdb.ca/                                           | https://www.ncbi.nlm.nih.gov/pubmed/29140435 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | HGNC                            | https://www.genenames.org/                                 | https://www.ncbi.nlm.nih.gov/pubmed/30304474 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | GwasCatalog                     | https://www.ebi.ac.uk/gwas/                                | https://www.ncbi.nlm.nih.gov/pubmed/30445434 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | FooDB                           | https://foodb.ca/                                          |                                              |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | DrugBank                        | https://www.drugbank.ca/                                   | https://www.ncbi.nlm.nih.gov/pubmed/29126136 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | DisGeNET                        | https://www.disgenet.org/                                  | https://www.ncbi.nlm.nih.gov/pubmed/25877637 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | DISEASES                        | https://diseases.jensenlab.org/                            | https://www.ncbi.nlm.nih.gov/pubmed/25484339 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | DGIdb                           | http://www.dgidb.org/                                      | https://www.ncbi.nlm.nih.gov/pubmed/29156001 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | CORUM                           | https://mips.helmholtz-muenchen.de/corum/                  | https://www.ncbi.nlm.nih.gov/pubmed/30357367 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Database    | Cancer Genome Interpreter       | https://www.cancergenomeinterpreter.org/                   | https://www.ncbi.nlm.nih.gov/pubmed/29592813 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Disease Ontology                | https://disease-ontology.org/                              | https://www.ncbi.nlm.nih.gov/pubmed/30407550 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Brenda Tissue Ontology          | https://www.brenda-enzymes.org/ontology.php?ontology_id=3  | https://www.ncbi.nlm.nih.gov/pubmed/25378310 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Experimental Factor Ontology    | https://www.ebi.ac.uk/efo/                                 | https://www.ncbi.nlm.nih.gov/pubmed/20200009 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Gene Ontology                   | http://geneontology.org/                                   | https://www.ncbi.nlm.nih.gov/pubmed/27899567 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Human Phenotype Ontology        | https://hpo.jax.org/                                       | https://www.ncbi.nlm.nih.gov/pubmed/27899602 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | SNOMED-CT                       | http://www.snomed.org/                                     | https://www.ncbi.nlm.nih.gov/pubmed/27332304 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Protein Modification Ontology   | https://www.ebi.ac.uk/ols/ontologies/mod                   | https://www.ncbi.nlm.nih.gov/pubmed/23482073 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Molecular Interactions Ontology | https://www.ebi.ac.uk/ols/ontologies/mi                    | https://www.ncbi.nlm.nih.gov/pubmed/23482073 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Mass Spectrometry Ontology      | https://www.ebi.ac.uk/ols/ontologies/ms                    | https://www.ncbi.nlm.nih.gov/pubmed/23482073 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
| Ontology    | Units Ontology                  | https://bioportal.bioontology.org/ontologies/UO            | https://www.ncbi.nlm.nih.gov/pubmed/23060432 |
+-------------+---------------------------------+------------------------------------------------------------+----------------------------------------------+
