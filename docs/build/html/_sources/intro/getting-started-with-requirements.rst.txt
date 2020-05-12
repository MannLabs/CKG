
Getting Started with some Requirements
========================================

The following instructions on installation of software requirements and setting up the Clinical Knowledge graph, are optimized for operating systems MacOS and Linux. For more detailed instructions on how to set up the CKG in Windows, pleaso go to :ref:`Windows installation`.


Java
-------

Before starting setting up Neo4j and, later on, the Clinical Knowledge Graph, it is very important that you have *Java* installed in your machine, including **Java SE Runtime Environment**.

Different versions of a Neo4j database can have different requirements. For example, Neo4j 3.5 versions require Oracle Java 8, while Neo4j 4.0 versions already require Oracle Java 11.
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
-----------

Another essential package for the functioning of the Clinical Knowledge Graph is R.

Make sure you have installed **R version >= 3.5.2**:

.. code-block:: bash

	$ R --version

And that R is installed in ``/usr/local/bin/R``:
	
.. code-block:: bash
	
	$ which R

To install the neccessary R packages, simply initiate R (terminal or shell) and run:

.. code-block:: python
	
	install.packages('BiocManager')
	BiocManager::install()
	BiocManager::install(c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute'))
	install.packages(c('flashClust','WGCNA', 'samr'), dependencies=TRUE, repos='http://cran.rstudio.com/')


.. note:: If you need to install R, follow `these <https://web.stanford.edu/~kjytay/courses/stats32-aut2018/Session%201/Installation%20for%20Mac.html>`__ tutorial.

.. warning:: In Mac OS, make sure you have **XQuartz** installed.

Now that you are all set, you can move on and start with Neo4j.