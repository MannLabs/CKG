Contributing
======================

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

* Types of Contributions `Types of Contributions`_
* Contributor Setup `Setting Up the Code for Local Development`_
* Contributor Guidelines `Contributor Guidelines`_
* Core Committer Guide `Core Committer Guide`_

Types of Contributions
----------------------------

You can contribute in many ways:

Create Analysis or Visualization methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you develop new ways of analysing or visualizing data, please feel free to add to the Analytics Core.

Report Bugs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Report bugs at `https://github.com/MannLabs/CKG/issues <https://github.com/MannLabs/CKG/issues>`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* If you can, provide detailed steps to reproduce the bug.
* If you don't have steps to reproduce the bug, just note your observations in as much detail as you can. Questions to start a discussion about the issue are welcome.

Fix Bugs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Look through the GitHub issues for bugs. Anything tagged with "bug" is open to whoever wants to implement it.

Implement Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Look through the GitHub issues for features. Anything tagged with "enhancement" and "please-help" is open to whoever wants to implement it.

Please do not combine multiple feature enhancements into a single pull request.

Write Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Clinical Knowledge Graph could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

If you want to review your changes on the documentation locally, you can do:

.. code-block:: bash

    $ cd docs/
    $ make servedocs


This will compile the documentation, open it in your browser and start watching the files for changes, recompiling as you save.

Submit Feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The best way to send feedback is to file an issue at `https://github.com/MannLabs/CKG/issues <https://github.com/MannLabs/CKG/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

Setting Up the Code for Local Development
-------------------------------------------

Here's how to set up ``CKG`` for local development.

1. Fork the ``CKG`` repo on GitHub.
2. Clone your fork locally:

.. code-block:: bash

    $ git clone git@github.com:MannLabs/CKG.git


3. Install your local copy according to the "Getting Started" tutorials.

4. Create a branch for local development:

.. code-block:: bash

    $ git checkout -b name-of-your-bugfix-or-feature


Now you can make your changes locally.

5. When you're done making changes, commit your changes and push your branch to GitHub:

.. code-block:: bash

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature


7. Submit a pull request through the GitHub website.

Contributor Guidelines
------------------------------

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you submit a pull request, check that it meets these guidelines:

1. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and describe it.
2. The pull request should work for Python 3.5, 3.6 and 3.7.

Coding Standards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* PEP8
* Functions over classes except in tests
* Quotes via `http://stackoverflow.com/a/56190/5549 <http://stackoverflow.com/a/56190/5549>`_

  * Use double quotes around strings that are used for interpolation or that are natural language messages
  * Use single quotes for small symbol-like strings (but break the rules if the strings contain quotes)
  * Use triple double quotes for docstrings and raw string literals for regular expressions even if they aren't needed.
  * Example:

.. code-block:: python

    LIGHT_MESSAGES = {
        'English': "There are %(number_of_lights)s lights.",
        'Pirate':  "Arr! Thar be %(number_of_lights)s lights."
    }
    def lights_message(language, number_of_lights):
        """Return a language-appropriate string reporting the light count."""
        return LIGHT_MESSAGES[language] % locals()
    def is_pirate(message):
        """Return True if the given message sounds piratical."""
        return re.search(r"(?i)(arr|avast|yohoho)!", message) is not None


* Write new code in Python 3.

Core Committer Guide
-------------------------

Vision and Scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Core committers, use this section to:

* Guide your instinct and decisions as a core committer
* Limit the codebase from growing infinitely

Command-Line and API Accessible
"""""""""""""""""""""""""""""""""""""

* Provides command-line utilities that launch a dash app to browse projects, statistics and others, create new users, and import and load data into the database.
* Extremely easy to use without having to think too hard
* Flexible for more complex use via optional arguments

Extensible
"""""""""""""""""""""""""""""""""""""

Being extendable by people with different ideas.

* Entirely function-based
* Aim for statelessness
* Lets anyone write more opinionated tools

Freedom for CKG users to build and extend.

* Community-based project, all contributions to improve and/or extend the code are welcome.

Inclusive
"""""""""""""""""""""""""""""""""""""

* Cross-platform support.
* Fixing Windows bugs even if it's a pain, to allow for use by the entire community.

Process: Pull Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a pull request is untriaged:

* Look at the roadmap
* Set it for the milestone where it makes the most sense
* Add it to the roadmap

How to prioritize pull requests, from most to least important:

* Fixes for broken code. Broken means broken on any supported platform or Python version.
* Features.
* Bug fixes.
* Major edits to docs.
* Extra tests to cover corner cases.
* Minor edits to docs.

Ensure that each pull request meets all requirements in `checklist <https://gist.github.com/audreyr/4feef90445b9680475f2>`_.

Process: Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If an issue is a bug that needs an urgent fix, mark it for the next patch release.
Then either fix it or mark as please-help.

For other issues: encourage friendly discussion, moderate debate, offer your thoughts.

New features require a +1 from 2 other core committers (besides yourself).

Process: Pull Request merging and HISTORY.md maintenance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you merge a pull request, you're responsible for updating ``AUTHORS.rst`` and ``HISTORY.rst``

When you're processing the first change after a release, create boilerplate following the existing pattern:

.. code-block:: text

    ## x.y.z (Development)

    The goals of this release are TODO: release summary of features

    Features:

    * Feature description, thanks to [@contributor](https://github.com/contributor) (#PR).

    Bug Fixes:

    * Bug fix description, thanks to [@contributor](https://github.com/contributor) (#PR).

    Other changes:

    * Description of the change, thanks to [@contributor](https://github.com/contributor) (#PR).


Process: Accepting New Features Pull Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Run the feature to generate the output.
* Attempt to include it in the standard pipeline and run an example project dataset.
* Merge the feature in.
* Update the history file.

note: Adding features doesn't give authors credit.

Process: Your own code changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All code changes, regardless of who does them, need to be reviewed and merged by someone else.
This rule applies to all the core committers.

Exceptions:

* Minor corrections and fixes to pull requests submitted by others.
* While making a formal release, the release manager can make necessary, appropriate changes.
* Small documentation changes that reinforce existing subject matter. Most commonly being, but not limited to spelling and grammar corrections.

Responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Ensure cross-platform compatibility for every change that's accepted. Windows, Mac, Debian & Ubuntu Linux.
* Ensure that code that goes into core meets all requirements in this checklist: `https://gist.github.com/audreyr/4feef90445b9680475f2 <https://gist.github.com/audreyr/4feef90445b9680475f2>`_
* Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
* Keep feature versions as small as possible, preferably one new feature per version.
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. Look at `Code of Conduct :ref:code-of-conduct`.
