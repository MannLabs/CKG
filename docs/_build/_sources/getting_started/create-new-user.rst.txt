Create a new user in the graph database
=======================================

The creation of a new user includes two steps:

1. The user who is currently logged-in in the database and invoking the commands, has to add the new user to the system and attribute it a role, by default ``reader``.

2. Each user is added in the graph database as a new ``User`` node, with attributes: id, username, name, acronym, email, secondary email, phone number, affiliation, rolename and expiration date.

There are multiple ways to create a new user:

**From the command line:** *(one user at a time)*

.. prompt:: bash $

	cd graphdb_builder/builder
	python create_user.py -u username -d password -n name -e email -s second_email -p phone_number -a affiliation

**From an excel file:** *(multiple users)*

.. prompt:: bash $

	cd graphdb_builder/builder
	python create_user.py -f path/to/excel/file

For help on how to use ``create_user.py``, run:

.. prompt:: bash $

	python create_user.py -h



For each new user, an access password is set be the same the the ``username``. All users are advised
to modify the default password as soon as they are granted access.

