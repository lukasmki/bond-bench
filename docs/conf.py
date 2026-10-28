# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bond-bench"
copyright = "2026, Lukas Kim"
author = "Lukas Kim"
release = _version("bond-bench")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
# Optional extras (``--extra ml`` / ``--extra plot``) need not be installed to build the docs.
autodoc_mock_imports = ["torch", "matplotlib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"bond-bench {release}"
html_static_path = ["_static"]
