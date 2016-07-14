SHELL := /bin/bash

help:
	@echo "lint - check code style with flake8"
	@echo "test - run tests quickly"
	@echo "coverage - check code coverage quickly"
	@echo "install_py2 - Get and install Miniconda Python2"
	@echo "install_py3 - Get and install Miniconda Python3"
	@echo "_miniconda - Internal utility function Install and update miniconda"

test:
	cp testing/matplotlibrc .
	py.test
	rm matplotlibrc

coverage:
	cp testing/matplotlibrc .
	coverage run --source anchor --omit=tests --module py.test
	rm matplotlibrc

lint:
	flake8 anchor

get_py2:
	wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
	make miniconda
	export PATH=~/miniconda2/bin:$PATH
	conda update --yes conda

get_py3:
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.s
	make miniconda
	export PATH=~/miniconda3/bin:$PATH
	conda update --yes conda


install_py2:
	# --- Check if "conda" command exists. if not, download miniconda and install it
	command -v conda >/dev/null 2>&1 || { make get_py2; }

	# --- Get proper python version
	if test -z "$$TRAVIS_PYTHON_VERSION"; then export ANCHOR_PYTHON_VERSION=2.7; echo "Specifying Python2.7"; else export ANCHOR_PYTHON_VERSION="$$TRAVIS_PYTHON_VERSION"; echo "Travis CI Python"; fi ;\
	echo ANCHOR_PYTHON_VERSION is $$ANCHOR_PYTHON_VERSION; \
	\
	# --- Create anchor environment \
	conda create -n anchor_py2 --yes python=$$ANCHOR_PYTHON_VERSION pip

	# --- Activate environment
	source activate anchor_py2

	# --- Install conda requirements first, then the rest by pip
	conda install --yes --file conda_requirements.txt
	pip install -r requirements.txt

	# --- Install anchor itself
	pip install .

install_py3:
	command -v conda >/dev/null 2>&1 || { make get_py3; }
	if [ -z $${TRAVIS_PYTHON_VERSION+x} ]; then ANCHOR_PYTHON_VERSION=3.5; else ANCHOR_PYTHON_VERSION="$$TRAVIS_PYTHON_VERSION"; fi
	conda create -n anchor_py3 --yes python="$$ANCHOR_PYTHON_VERSION" pip
	source activate anchor_py3
	conda install --yes pytest
	conda install --yes --file conda_requirements.txt
	pip install -r requirements.txt
	pip install .

miniconda:
	chmod +x miniconda.sh
	"./miniconda.sh -b"
