# Makefile for the NHS Patient Feedback Intelligence App.
#
# Usage:
#   make install   # install pinned dependencies into the active environment
#   make data      # generate the synthetic NHS-style dataset
#   make train     # build models/model.pkl from data/processed/
#   make lint      # run ruff
#   make test      # run pytest
#   make app       # launch the Streamlit app
#   make clean     # remove generated artefacts

PYTHON ?= python

.PHONY: install data train lint test app clean all

all: data train lint test

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) data/make_dataset.py
	$(PYTHON) data/preprocess.py

train:
	$(PYTHON) train_model.py

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

app:
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf models/model.pkl
	rm -rf data/processed/*.csv
	rm -rf mlruns
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
