SHELL := /bin/bash

help:
	@echo "lint - check code style with flake8"
	@echo "test - run tests quickly"
	@echo "coverage - check code coverage quickly"
	@echo "install - Maybe get Miniconda, and install this package into an environment"
	@echo "    using conda. Requires 'PYTHON_VERSION' argument (see example below)"
	@echo "  Example:"
	@echo "    'make PYTHON_VERSION=2.7 install' will install anchor into an environment"
	@echo "    called 'anchor_py2.7', cowardly not overwriting any environment that"
	@echo "    existed there before."

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
	# http://conda.pydata.org/docs/travis.html#the-travis-yml-file
	wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
	bash miniconda.sh -b -p $$HOME/miniconda
	export PATH="$HOME/miniconda/bin:$PATH"
	hash -r
	conda config --set always_yes yes --set changeps1 no
	conda update -q conda
	conda info -a

install:
	# --- Create anchor environment
	conda create -n anchor_py${PYTHON_VERSION} --yes python=${PYTHON_VERSION} pip
	\
	# Update conda again
	conda update conda \
	\
	# --- Activate environment
	source activate anchor_py${PYTHON_VERSION} \
	\
	# --- Install conda requirements first, then the rest by pip \
	conda install --yes --file conda_requirements.txt \
	pip install -r requirements.txt \

	# --- Install anchor itself
	pip install .
