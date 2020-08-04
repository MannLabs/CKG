# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath( './../..'))

# MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()
autodoc_mock_imports = ['pandas', 'numpy', 'scipy', 'matplotlib', 'h5py', 'rpy2', 'sklearn', 'lifelines', 'autograd', 'umap', 'numba']

# -- Project information -----------------------------------------------------

project = 'ClinicalKnowledgeGraph'
copyright = '2019, Alberto Santos, Ana Rita Colaço, Annelaura B. Nielsen'
author = 'Alberto Santos, Ana Rita Colaço, Annelaura B. Nielsen'

# The short X.Y version
version = '1.0b1 BETA'

# The full version, including alpha/beta/rc tags
release = '1.0b1 BETA'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinxcontrib.httpdomain',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    # 'sphinx-prompt',
    'recommonmark',
    # 'notfound.extension',
]

autosectionlabel_prefix_document = True

## Include Python objects as they appear in source files
## Default: alphabetically ('alphabetical')
autodoc_member_order = 'bysource'
## Default flags used by autodoc directives
autodoc_default_options = {'members': True,
                            'show-inheritance': True,
                            'undoc-members': True}
## Generate autodoc stubs with summaries from code
autosummary_generate = True

# The suffix of source filenames.
source_suffix = {'.rst': 'restructuredtext',}

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['Thumbs.db', '.DS_Store']



# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Suppress nonlocal image warnings
suppress_warnings = ['image.nonlocal_uri']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
## alabater theme opitons
# html_theme_options = {
#     'github_button': True,
#     'github_type': 'star&v=2',  ## Use v2 button
#     'github_user': 'romanvm',
#     'github_repo': 'sphinx_tutorial',
#     'github_banner': True,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
## Sidebars configuration for alabaster theme

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

## I don't like links to page reST sources
html_show_sourcelink = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'ClinicalKnowledgeGraphDocs'
language = 'en'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
'preamble': '',

# Latex figure (float) alignment
'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'ClinicalKnowledgeGraphdocs.tex', 'Clinical Knowledge Graph Documentation',
   'Alberto Santos, Ana Rita Colaço, Annelaura B. Nielsen', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'ckgdocs', 'Clinical Knowledge Graph Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'ClinicalKnowledgeGraphdocs', 'Clinical Knowledge Graph Documentation',
   author, 'ClinicalKnowledgeGraphdocs', 'One line description of project.',
   'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/3.6/': None}


# A string of reStructuredText that will be included at the end of every source
# file that is read.
rst_epilog = """

"""
