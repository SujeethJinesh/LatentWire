.PHONY: install test calibrate evaluate control ablations

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

calibrate:
	$(PYTHON) scripts/calibrate.py --help

evaluate:
	$(PYTHON) scripts/evaluate.py --help

control:
	$(PYTHON) scripts/run_control_suite.py --help

ablations:
	$(PYTHON) scripts/ablation_sweep.py --help
