
.. _Windows installation:

Getting started with Windows
===============================

In this section we describe how to install all the necessary requirements and set up the Clinical Knowledge Graph on a Windows operating system.


Java
-------

Similarly to MacOS and Linux, Windows will also need a **Java** installation (Java SE Runtime Environment and Java SE Development Kit).

Be aware that different versions of a Neo4j database can have different requirements. For example, Neo4j 3.5 versions require Oracle Java 8, while Neo4j 4.0 versions already require Oracle Java 11.
When using a new version of Neo4j, always remember to read the respective Operations Manual, and check for the software requirements.

By default Java should be installed on the Windows 10. If this is not your case, please follow this `tutorial <https://docs.oracle.com/javase/8/docs/technotes/guides/install/windows_jdk_install.html#A1097936>`__ to install it.


Neo4j
-------

The installation of Neo4j on Windows follows the same steps as :ref:`Getting Started with Neo4j`:

1. Download a copy of the Neo4j desktop version from the `Neo4j download page <https://neo4j.com/download/>`__.

#. Install Neo4j by following the instructions automatically opened in the browser.

#. Open the Neo4j Desktop App and create a database by clicking :guilabel:`Add graph`, followed by :guilabel:`Create a Local Graph`, using the password "NeO4J".

#. Click :guilabel:`Manage` and then :guilabel:`Plugins`. Install "**APOC**" and "**GRAPH ALGORITHMS**".

#. Click the tab :guilabel:`Settings`, and comment the option ``dbms.directories.import=import`` by adding ``#`` at the beginning of the line.

#. Click :guilabel:`Apply` at the bottom of the window.

#. Start the Graph by clicking the play sign, at the top of the window.

To check for errors, please go to tab :guilabel:`Logs`.


Python
----------

Installing Python 3.6.8 is an essential part of setting up the CKG.
To do so, simply use the `link <https://www.python.org/downloads/windows/>`__ to download a python 3.6 version executable installer (e.g. Python 3.6.8 - Dec. 24, 2018).
Follow the steps and select the option ``Add Python 3.6 to PATH``.

Once finished, you can check if python was successfully added to PATH, by opening the environment variables window:

1. Go to the Windows menu, right-click on :guilabel:`Computer` and click on :guilabel:`Properties`.

#. From the computer properties dialog, select ``Advanced system settings`` on the left panel. And from there, click on :guilabel:`Environment variables` button.

#. In the top half of the new window (**User variables**), look for a variable with the name ``Path``. If you select it and click :guilabel:`Edit`, the full path to Python36 should be included in the variable value. Besides this, you can also add the paths to python 3.6 ``site-packages`` and ``Scripts`` folders, to the ``Path`` variable.

#. When you are done, click :guilabel:`OK` to save, and click :guilabel:`OK` in all the opened windows.


Microsoft Visual C++ Build Tools
----------------------------------

Running python on Windows can sometimes result in the following error:

.. code-block:: python

	error Microsoft Visual C++ 14.0 is required

To fix this error, you will need to download and install Microsoft `Build Tools for Visual Studio <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`__.

Once installed, click ``Workloads``, select all the packages available and install them. This will require several Gigabytes of disk space so, as an alternative and if your machine has limited space, you can install only ``C++ build tools`` under ``Workloads``, and ``Windows 10 SDK`` and the latest version of ``MSVC v142 - VS 2019 C++ x64/x86 build tools`` under ``Individual Components``.

The build tools allow Python packages to be built in Windows, from the command line (MSVC cl.exe module is used as a C/C++ compiler).


R
-------

Another essential package for the functioning of the Clinical Knowledge Graph is R.

You can check if an **R version >= 3.5.2** is already installed by running:

.. code-block:: bash

	> where R

If R is not installed in your machine, please follow `these tutorial <https://rstudio-education.github.io/hopr/starting.html>`__.

In order to simplify calling R from the command prompt, you can choose to add it to ``PATH`` and to the environment variables. To do so, follow the steps bellow:

1. Go to the Windows menu, right-click on :guilabel:`Computer` and click on :guilabel:`Properties`.

#. From the computer properties dialog, select ``Advanced system settings`` on the left panel. And from there, click on :guilabel:`Environment variables` button.

#. In the Environment variables dialog, click the :guilabel:`New` button in the top half of the dialog, to make a new user variable.

#. Give the variable name as ``R`` and the value is the path to the R executable, which is usually ``C:\Program Files\R\R-4.0.0\bin\R.exe``.

#. In the bottom half of the Environment variables dialog, find the variable ``Path``, select it and click :guilabel:`Edit`.

#. In the edit dialog window, add ``;`` to the end of the variable value followed by the R path used when creating the previous environmental variable.

#. Click :guilabel:`OK` to save, click :guilabel:`OK` and :guilabel:`OK` again to save the new variable and edit to ``Path``.


To confirm that the environment variable is correctly set in command line type:

.. code-block:: bash

	> echo %R%


This will print the path you used as value (e.g. ``C:\Program Files\R\R-4.0.0\bin\R.exe``).

To run R from the command prompt, run:

.. code-block:: bash

	> R


All R packages can be installed by simply initiating R (command prompt or R shell) and running:

.. code-block:: python

	install.packages('BiocManager')
	BiocManager::install()
	BiocManager::install(c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute'))
	install.packages(c('flashClust','WGCNA', 'samr'), dependencies=TRUE, repos='http://cran.rstudio.com/')

.. warning:: If the install does not work (cannot write to library), run a new command prompt as administrator:

1. Go to the Windows menu, right-click on :guilabel:`Command Prompt` and select ``Run as administrator``.

In this new prompt, launch R and run the previous R install packages.


Getting Started with the CKG Build
------------------------------------

Setting up the Clinical Knowledge Graph is thoroughly described here.
Assuming you have **Python 3.6** already installed, you can choose to create a virtual environment where all the packages with the specific versions will installed.

To check which Python version is currently installed, run in the command prompt (cmd.exe):

.. code-block:: bash

	> python --version

And to find your this Python version is installed:

.. code-block:: bash

	> where python


Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a new Python virtual environment, you can choose to use ``virtualenv`` or ``venv``.

The usage of ``virtualenv`` is exemplified in ref:`Create virtual environment`.

To use ``venv``, open a command prompt (cmd.exe) window and type:

.. code-block:: bash

	> python -m venv path\to\env_name

.. note:: ``path\to\env_name`` should be replaced with the relative or full path to where you want to place your virtual environment, while the ``env_name`` part is to be replaced with the name you want to attribute to the virtual environment.

Whichever way you create the virtual environment, the activation method is the same:

.. code-block:: bash

	> path\to\env_name\Scripts\activate.bat


After this, the name of the virtual environment will now appear on the left of the prompt:

.. code-block:: bash

	(env_name) C:\>

If you are finished working in the virtual environment for the moment, you can deactivate it by running:

.. code-block:: bash

	> deactivate

.. warning:: Remember, every time you are working with the CKG, the virtual environment needs to be activated first.



Setting up the CKG
^^^^^^^^^^^^^^^^^^^^^^^^

Once you have cloned the master branch of the CKG GitHub repository, all the Python packages necessary to run the Clinical Knowledge Graph can be found in ``requirements.txt``.

Unfortunately, due to incompatibilities of the current versions ``celery`` and ``rpy2`` packages need to be removed from ``requirements.txt`` before installing all other packages.

To do so, open the mentioned file in your preferred text editor tool (e.g. Notepad) and add ``#`` in the beginning of the lines ``celery==4.3.0`` and ``rpy2==3.0.5``. Save and close the file, making sure it is saved as a plain text file.

.. warning:: Part of the CKG functionality includes interfacing Python and R, and seemingly use R functions for data analysis. The python package ``rpy2`` is used as this interface and unfortunately, the current release of this package for Windows is not compatible with CKG. Installation of the CKG on Windows machines, will therefore **not** allow the usage of R packages (SAMR and WGCNA) within the CKG.


To install all the required packages, simply run:

.. code-block:: bash

	> cd CKG\
	> pip3 install --upgrade pip
	> pip3 install --ignore-installed -r requirements.txt

	.. warning:: Make sure the virtual environment previously created is active before installing ``requirements.txt``.

Now that all the packages are correctly installed, you will have to create the appropriate directory architecture within the local copy of the cloned repository:

.. code-block:: bash

	> python setup_CKG.py
	> python setup_config_files.py

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where all errors and warnings related to the code will be written to.


Add CKG to environmental variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to run CKG modules, the package needs to be added to the environmental variables.


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
