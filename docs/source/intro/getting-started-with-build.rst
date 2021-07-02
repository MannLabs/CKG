.. _Installing CKG python library:

.. include:: ../global.rst

Installing CKG python library
===================================

Setting up the Clinical Knowledge Graph is straightforward.
Assuming you have **Python** |python_version| already installed and a virtual environment created following instructions here: :ref:`Installation`.



Setting up the Clinical Knowledge Graph
-----------------------------------------

The first step in setting up the CKG, is to obtain the complete code by cloning the GitHub repository:

.. code-block:: bash

	$ git clone https://github.com/MannLabs/CKG.git

Another option is to download it from the github page directly:

1. Go to https://github.com/MannLabs/CKG

2. In `Code` select **Download ZIP**

3. Unzip the file

Once this the cloning is finished or the file is unzipped, you can install CKG by running:

.. code-block:: bash

	$ cd CKG/
	$ conda activate ckgenv
	$ python setup.py install

This will automatically create the ``data`` folder and all subfolders, as well as setup the configuration for the log files where all errors and warnings related to the code will be written to.
Further, it will create an executable file with CKG's app. To start the app, simpy run:

.. code-block:: bash

	$ ckg_app


.. warning:: If you are using a Unix Operating System (i.e MacOS or Linux), you will need to start Redis server by running:
			$ redis-server

