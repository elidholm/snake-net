SHELL := /bin/bash

fmt:
	black .
	isort **/*.py

test:
	pytest .

check:
	mypy .

lint:
	flake8 .
	pylint .
