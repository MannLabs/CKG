PYPI_SERVER = pypitest

define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

.DEFAULT_GOAL := help

.PHONY: lint
lint: ## Check code style with flake8
	@echo "+ $@"
	@tox -e flake8

.PHONY: docs
docs: ## Generate Sphinx HTML documentation, including API docs
	@echo "+ $@"
	@rm -f docs/src.rst
	@sphinx-apidoc -o docs/ src
	@rm -f docs/modules.rst
	@$(MAKE) -C docs clean
	@$(MAKE) -C docs html
	@$(BROWSER) docs/_build/html/index.html

.PHONY: servedocs
servedocs: docs ## Rebuild docs automatically
	@echo "+ $@"
	@watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

.PHONY: submodules
submodules: ## Pull and update git submodules recursively
	@echo "+ $@"
	@git pull --recurse-submodules
	@git submodule update --init --recursive

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
