# postCKG

## Contents

file                      | description
------------------------- | --------------------------------------
[postCKG](postCKG.ipynb)    | Contains notebook of accessing and extracting from CKG output files (proteomics folder), drug target prioritization, clustergrammer2 visualization
[environment.yml](environment.yml)    | environment.yml file
[requirements.txt](requirements.txt)    | requirements.txt file

## Instructions
To be able to run the notebook, best would be to create a virtual environment and install the dependcies, packages needed for this notebook by typing in terminal/command line the following code:

```
conda env create -f environment.yml
conda install --file requirements.txt
```

In case certain packages not found in the default channel, try pip install individual packages

You need to install ipywidgets (https://ipywidgets.readthedocs.io/en/stable/user_install.html)

If you are using jupyterLab, you need to install the ipywidgets JupyterLab extension by typing in terminal the following code:
```
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
You would also need the proteomics folder from CKG output results. 
