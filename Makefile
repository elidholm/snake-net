SHELL := /bin/bash

ci: fmt test check lint

test:
	pytest .

check:
	mypy .

fmt: black isort

black:
	black .

isort:
	isort **/*.py

lint: flake8 pylint

flake8:
	flake8 .

pylint:
	pylint .

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf **/__pycache__
