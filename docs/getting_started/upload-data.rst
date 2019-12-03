Upload project experimental data
================================

Prepare data for upload
-----------------------

**Clinical Data**

Open the 'Clinical Data template' in Excel and fill in as much information as you can.
Be aware that it is mandatory to fill the columns:

- subject external_id	tissue
  *This is the id your subject has in your study so far*
- disease
  *This should be identical to the disease you selected from the drop-down menu in the 'Project creation'_*
- biological_sample external_id
  *This is the id of the sample taken from your subject, if you have both blood and urine for every subject, you should correspondingly have two biological sample ids for each subject id*
- biological_sample quantity
  *Amount of biological sample*
- biological_sample quantity_units
  *Unit*
- analytical_sample external_id
  *If multiple analyses were performed on the same biological sample, eg. proteomics and transcriptomics, there should be multiple analytical sample id's for every biological sample*
- analytical_sample quantity
  *Amount of sample used in the experiment*
- analytical_sample quantity_units
  *Unit*
- grouping1
  *Annotate grouping of each sample*
- grouping2
  *If there are more than one grouping use this column to add a second level, for now only two groupings are possible*

Additional clinical information about your study subjects can be added in the subsequent columns.
Please use SNOWMED terms as headers for every new column you add. This will be used to gather existing information about the type of data you have.
Follow this link to search for 'SNOWMED'_ terms:
.. _SNOWMED: https://browser.ihtsdotools.org/?perspective=full&conceptId1=734000001&edition=MAIN/2019-07-31&release=&languages=en
Example: To add a column with "Age" search for "age". This gives multiple matches, with the first one being: "Age (qualifier value), SCTID:397669002". Please enter this information as column header with the SCTID in brackets: Age (qualifier value) (397669002)

If you can't find your header in SNOWMED you should contact annelaura.bach@cpr.ku.dk, put "Header Creation, CKG" in the subject.
In the email please provide your "missing" header and a description of what it is.

**Proteomic data**
Do not perform any imputations or similar on your data before uploading it. This will be carried out by the CKG.

Data from MaxQuant(DDA):
Upload the proteinGroups.txt
Upload

Data from Spectronaut(DIA):
?

You can proceed to 'Upload data'_ when you have prepared both your clinical and proteomics data.

Upload data
___________

Go to:
http://localhost:5000/apps/dataUpload/{YOUR_PROJECT_ID}

Upload your clinical data file. This may take a few minutes.
Select "Clinical" from the drop-down menu before you press "Upload"

Upload your proteomic data file. This may take a few minutes.
Select "Proteomics" from the drop-down menu before you press "Upload"
