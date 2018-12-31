init:
	pip install -U tox
	pip install -U isort

isort:
	pip install -U isort
	isort -rc causalimpact
	isort -rc tests

isort-check:
	isort -ns __init__.py -rc -c -df causalimpact tests

flake8:
	pip install -U flake8
	flake8 causalimpact tests

coverage:
	python setup.py test --coverage=true

test:
	python setup.py test 

publish:
	pip install -U setuptools
	pip install -U wheel
	pip install 'twine>=1.5.0'
	python setup.py sdist bdist_wheel
	twine upload dist/*
	rm -fr build dist .egg pycausalimpact.egg-info

.PHONY: flake8 isort coverage test publish isort-check
