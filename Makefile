install:
	python setup.py install

dist: clean
	python setup.py sdist bdist_wheel

clean-pyc:
	find . -name '*.pyc' -type f -exec rm {} +
	find . -name '*.pyo' -type f -exec rm {} +
	find . -name '__pycache__' -type d -exec rm -r {} +

clean-build:
	rm -rf build/
	rm -rf dist/

clean: clean-pyc clean-build

format:
	yapf --parallel --in-place --recursive --exclude 'pydra/_version.py' pydra

lint:
	flake8

test: clean-pyc
	py.test -vs -n auto --cov pydra --cov-config .coveragerc --cov-report xml:cov.xml --doctest-modules pydra
