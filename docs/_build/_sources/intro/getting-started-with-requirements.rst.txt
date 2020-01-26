
Getting Started with some Requirements
========================================

Java
-------

Before starting setting up Neo4j and, later on, the Clinical Knowledge Graph, it is very important that you have *Java* installed in your machine.

To check if you already have **Java SE Runtime Environment 1.8** installed, run ``/usr/libexec/java_home -v 1.8`` in the terminal. This should print out
a path like ``/Library/Java/JavaVirtualMachines/jdk1.8.0_171.jdk/Contents/Home``. If you got this message ``Unable to find any JVMs matching version "1.8".``, 
please follow the steps below:

1. Go to ``https://www.oracle.com/technetwork/java/javase/downloads/jre8-downloads-2133155.html`` and download the version that fits to your OS.

2. Install the package.

3. Run ``/usr/libexec/java_home -v 1.8`` in the terminal to make sure the *Java* package has been installed in ``/Library/Java/JavaVirtualMachines/``.


R 
-----------

Another essential package for the functioning of the Clinical Knowledge Graph is R.

Make sure you have installed **R version 3.5.2**:

.. prompt:: bash $

	R --version

And that R is installed in ``/usr/local/bin/R``:
	
.. prompt:: bash $
	
	which R

To install the neccessary R packages, simply initiate R (terminal or shell) and run:

.. code-block:: python
	
	install.packages('BiocManager')
	BiocManager::install()
	BiocManager::install(c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute'))
	install.packages(c('flashClust','WGCNA', 'samr'), dependencies=TRUE, repos='http://cran.rstudio.com/')


.. note:: If you need to install R, follow `these <https://web.stanford.edu/~kjytay/courses/stats32-aut2018/Session%201/Installation%20for%20Mac.html>`_ tutorial.

Now that you are all set, you can move on and start with Neo4j.