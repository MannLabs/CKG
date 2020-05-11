
.. _Windows installation:

Getting started with Windows
===============================

In this section we describe how to install all the neccessary requirements and set up the Clinical Knowledge Graph on a Windows operating system.


Java
-------

Similarly to MacOS and Linux, Windows will also need a **Java** installation.
Be aware that different versions of a Neo4j database can have different requirements. For example, Neo4j 3.5 versions require Oracle Java 8, while Neo4j 4.0 versions already require Oracle Java 11.
When using a new version of Neo4j, always remember to read the respective Operations Manual, and check for the software requirements.

To check if you already have **Java SE Development Kit** installed, run ``java -version`` in your terminal window. This should print out three lines similar to the following, with possible variation in the version:

.. code-block:: python
	
	java version "1.8.0_171"
	Java(TM) SE Runtime Environment (build 1.8.0_171-b11)
	Java HotSpot(TM) 64-Bit Server VM (build 25.171-b11, mixed mode)

Running ``/usr/libexec/java_home`` in the terminal should print out a path like ``/Library/Java/JavaVirtualMachines/jdk1.8.0_171.jdk/Contents/Home``. Otherwise, please follow the steps below:

1. Go to ``https://www.oracle.com/java/technologies/javase-downloads.html`` and download the version that fits your Neo4j version and OS requirements.

#. Install the package.

#. Run ``/usr/libexec/java_home`` in the terminal to make sure the *Java* package has been installed in ``/Library/Java/JavaVirtualMachines/``.



R
-------

Another essential package for the functioning of the Clinical Knowledge Graph is R.

You can check if an **R version >= 3.5.2** is already installed by running:

.. code-block:: bash
	
	> where R

And all R packages can be installed by simply initiating R (command prompt or R shell) and running:

.. code-block:: python

	install.packages('BiocManager')
	BiocManager::install()
	BiocManager::install(c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute'))
	install.packages(c('flashClust','WGCNA', 'samr'), dependencies=TRUE, repos='http://cran.rstudio.com/')

If R is not installed in your machine, please follow `these tutorial <https://rstudio-education.github.io/hopr/starting.html>`__.


Neo4j
-------

The installation of Neo4j on Windows follows the same steps as :ref:`Getting Started with Neo4j`:

1. Download a copy of the Neo4j desktop version from the `Neo4j download page <https://neo4j.com/download/>`__.

#. Install Neo4j by following the instructions automatically opened in the browser.

#. Open the Neo4j Desktop App and create a database by clicking :guilabel:`Add graph`, followed by :guilabel:`Create a Local Graph`, using the password "bioinfo1112".

#. Click :guilabel:`Manage` and then :guilabel:`Plugins`. Install "**APOC**" and "**GRAPH ALGORITHMS**".

#. Click the tab :guilabel:`Settings`, and comment the option ``dbms.directories.import=import`` by adding ``#`` at the beginning of the line.

#. Click :guilabel:`Apply` at the bottom of the window.

#. Start the Graph by clicking the play sign, at the top of the window.

To check for errors, please go to tab :guilabel:`Logs`.


.. _Add Neo4j graph database to environmental variables:

Add Neo4j graph database to environmental variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add the Neo4j database you just created to your Windows environment variables:

1. Go to the Windows menu, right-click on :guilabel:`Computer` and click on :guilabel:`Properties`.

#. From the computer properties dialog, select ``Advanced system settings`` on the left panel. And from there, click on :guilabel:`Environment variables` button.

#. In the Environment variables dialog, click the :guilabel:`New` button in the top half of the dialog, to make a new user variable

#. Give the variable name as ``NEO4J_HOME`` and the value is the path to the previously created Neo4j database to the file, for example ``C:\Neo4J\neo4jDatabases\database-bab515f2-ffe7-4282-9bb5-648a53b8b566\installation-3.5.2\``

#. Click :guilabel:`OK` and :guilabel:`OK` again to save this variable.


To confirm that the environment variable is correctly set in command line type: 

.. code-block:: bash

	> echo %NEO4J_HOME% 


This will print the path you used as value (e.g. ``C:\Neo4J\neo4jDatabases\database-bab515f2-ffe7-4282-9bb5-648a53b8b566\installation-3.5.2\``).


.. warning:: Depending on your system, the path may vary. To check the path to the database go to ``Logs`` in the Neo4j Desktop interface.



Getting Started with the CKG Build
------------------------------------

Setting up the Clinical Knowledge Graph is thoroughly described here.
Assuming you have **Python 3.6** already installed, you can choose to create a virtual environment where all the packges with the specific versions will installed.

To check which Python version is currently installed, run in the command prompt (cmd.exe):

.. code-block:: bash

	> python --version

And to find your this Python version is installed:

.. code-block:: bash
	
	> where python


If you don't have **Python 3.6** installed in Windows, we recommend installing and using the Anaconda distribution of Python. Download at ``https://www.anaconda.com/products/individual#Downloads``, and follow the instructions from ``https://docs.anaconda.com/anaconda/install/windows/#``.


Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a new Python virtual environment, please use Anaconda conda installation.

1. In the command prompt (cmd.exe) type:

.. code-block:: bash

	> conda create -n env_name python=3.6

The key word ``python`` specifies which version of Python the virtual environment is requested to use, while ``env_name`` is the name of the virtual environment and can be set to anything you like.

2. Activate the virtual environment by running:

.. code-block:: bash

	> conda activate env_name

3. After this, the name of the virtual environment will now appear on the left of the prompt:

.. code-block:: bash

	(env_name) C:\>

4. If you are finished working in the virtual environment for the moment, you can deactivate it by running:

.. code-block:: bash
	
	> conda deactivate


.. warning:: Remember, everytime you are working with the CKG, the virtual environment needs to be activated first.



Setting up the CKG
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have cloned the master branch of the CKG GitHub repository, all the Python packages necessary to run the Clinical Knowledge Graph can be found in ``requirements.txt``.

Unfortunately, due to incompatibilities of the current Anaconda version of the ``rpy2`` package, this package needs to be removed from ``requirements.txt`` before installing all other packages.

To do so, open the mentioned file in your preferred text editor tool (e.g. Notepad) and remove the line ``rpy2==3.0.5``. Save and close the file, making sure it is saved as a plain text file.


.. warning:: Part of the CKG functionality includes interfacing Python and R, and seemingly use R functions for data analysis. The python package ``rpy2`` is used as this interface and unfortunately, the current release of this package for Windows is not compatible with CKG. Installation of the CKG on Windows machines, will therefore **not** allow the usage of R packages (SAMR and WGCNA) within the CKG.


To install all the required packages, simply run:

.. code-block:: bash
	
	> cd CKG/
	> conda install --file requirements.txt


.. warning:: Make sure the virtual environment previously created is active before installing ``requirements.txt``.


Now that all the packages are correctly installed, you will have to create the appropriate directory architecture within the local copy of the cloned repository:

.. code-block:: bash
	
	> python setup_CKG.py
	> python setup_config_files.py

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where all errors and warnings related to the code will be written to.


Add CKG to environmental variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly to :ref:`Add Neo4j graph database to environmental variables`, CKG also needs to be added to the environmental variables.


1. Go to the Windows menu, right-click on :guilabel:`Computer` and click on :guilabel:`Properties`.

#. From the computer properties dialog, select ``Advanced system settings`` on the left panel. And from there, click on :guilabel:`Environment variables`.

#. In the Environment variables dialog, click :guilabel:`New` in the top half of the dialog, to make a new user variable

#. Give the variable name as ``PYTHONPATH`` and the value is the path to the CKG code directory, for example ``C:\CKG\src``. Notice that the path should always finish with ``\CKG\src``.

#. Click :guilabel:`OK` and :guilabel:`OK` again to save this variable.


To confirm that the environment variable is correctly set in command line type: 

.. code-block:: bash

	> echo %PYTHONPATH% 

This will print the path you used as value (e.g. ``C:\CKG\src``).



Build Neo4j graph database (Windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building the CKG database in Windows follows the same steps as in MacOS and Linux so, from here on, please follow the tutorial :ref:`Build Neo4j graph database`.















