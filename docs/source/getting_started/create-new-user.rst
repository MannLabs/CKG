Create a new user in the graph database
=======================================

The creation of a new user includes two steps:

1. The user who is currently logged-in in the database and invoking the commands, has to add the new user to the system and attribute it a role, by default ``reader``.

#. Each user is added in the graph database as a new ``User`` node, with attributes: id, username, name, acronym, email, secondary email, phone number, affiliation, rolename and expiration date.

There are multiple ways to create a new user:

**From the command line:** *(one user at a time)*

.. code-block:: bash

	$ cd ckg/graphdb_builder/builder
	$ python create_user.py -u username -d password -n name -e email -s second_email -p phone_number -a affiliation

**From an excel file:** *(multiple users)*

.. code-block:: bash

	$ cd ckg/graphdb_builder/builder
	$ python create_user.py -f path/to/excel/file

For help on how to use ``create_user.py``, run:

.. code-block:: bash

	$ python create_user.py -h

.. warning:: If you want to have spaces (" ") in any of the arguments (e.g. -n name), you need to have the argument value within quotes "" (e.g. -n "John Smith"). The same applies to other arguments like affiliation.

**From an CKG's app:**

In CKG's app there is an Admin section that provides the option to create users in the database.

.. image:: ../_static/images/Admin_create_user.png