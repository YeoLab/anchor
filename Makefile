SHELL := /bin/bash

help:
	@echo "lint - check code style with flake8"
	@echo "test - run tests quickly"
	@echo "coverage - check code coverage quickly"
	@echo "install - Maybe get Miniconda, and install this package into an environment"
	@echo "    using conda. Requires 'PYTHON_VERSION' argument (see example below)"
	@echo "  Example:"
	@echo "    'make PYTHON_VERSION=2.7 install' will install anchor into an environment"
	@echo "    called 'anchor_py2.7'"

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

get_miniconda:
	wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
	chmod +x miniconda.sh
	"./miniconda.sh -b"
	export PATH=~/miniconda3/bin:$PATH
	conda update --yes conda


install:
	# --- Check if "conda" command exists. if not, download miniconda and install it
	command -v conda >/dev/null 2>&1 || { make get_miniconda; }

	# --- Create anchor environment
	conda create -n anchor_py${PYTHON_VERSION} --yes python=${PYTHON_VERSION} pip

	# --- Activate environment
	source activate anchor_py${PYTHON_VERSION}

	# --- Install conda requirements first, then the rest by pip
	conda install --yes --file conda_requirements.txt
	pip install -r requirements.txt

	# --- Install anchor itself
	pip install .
