.. image:: ./src/report_manager/assets/banner.jpg
    :align: center

**Clinical Knowledge Graph**  
============================
*version: 1.0b1* *BETA*

A Python project that allows you to analyse proteomics and clinical data, and integrate and mine knowledge from multiple biomedical databases widely used nowadays.

* Documentation: `https://CKG.readthedocs.io <https://CKG.readthedocs.io>`_

* GitHub: `https://github.com/MannLabs/CKG <https://github.com/MannLabs/CKG>`_
* Free and open source software: `MIT license <https://github.com/MannLabs/CKG/LICENSE.rst>`_
* Reference: https://www.biorxiv.org/content/10.1101/2020.05.09.084897v1


Abstract
------------

.. image:: ./src/report_manager/assets/abstract.png
    :align: center

Several omics data types are already used as diagnostic markers, beacons of possible treatment or prognosis. Advances in technology have paved the way for omics to move into the clinic by generating increasingly larger amounts of high-quality quantitative and qualitative data.  Additionally, knowledge around these data has been collected in diverse public resources, which has facilitated the understanding of these data to some extent. However, there are several challenges that hinder the translation of high-throughput omics data into identifiable, interpretable and actionable clinical markers. One of the main challenges is the interpretation of the multiple hits identified in these experiments. Furthermore, a single omics dimension is often not sufficient to capture the full complexity of disease, which would be aided by integration of several of them. To overcome these challenges, we propose a system that integrates multi-omics data and information spread across a myriad of biomedical databases into a Clinical Knowledge Graph (CKG).  This graph focuses on the data points or entities not as silos but as related components of a graph. To illustrate, in our system an identified protein in a proteomics experiment encompasses also all its related components (other proteins, diseases, drugs, etc.) and their relationships. Thus, our CKG facilitates the interpretation of data and the inference of meaning by providing relevant biological context. Further, ~. Here we describe the current state of the system and depict its use by applying it to use cases such as treatment decisions using cancer genomics and proteomics.


Cloning and installing
-----------------------

The setting up of the CKG includes several steps and might take a few hours (if you are building the database from scratch). However, we have prepared documentation and manuals that will guide through every step.
To get a copy of the GitHub repository on your local machine, please open a terminal windown and run:

.. code-block:: bash

	$ git clone https://github.com/MannLabs/CKG.git

This will create a new folder named "CKG" on your current location. To access the documentation, use the ReadTheDocs link above, or open the html version stored in the *CKG* folder `CKG/docs/build/html/index.html`. After this, follow the instructions in "First Steps" and "Getting Started".

.. warning:: If git is not installed in your machine, please follow this `tutorial <https://www.atlassian.com/git/tutorials/install-git>`__ to install it.


Features
---------------

* Cross-platform: Mac, and Linux are officially supported.

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
