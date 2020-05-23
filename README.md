# Pydra <img src="pydra_logo.png" width="50">
<!--(https://raw.githubusercontent.com/nipype/pydra/master/pydra_logo.jpg)-->

A simple dataflow engine with scalable semantics.

[![Build Status](https://travis-ci.org/nipype/pydra.svg?branch=master)](https://travis-ci.org/nipype/pydra)
[![CircleCI](https://circleci.com/gh/nipype/pydra.svg?style=svg)](https://circleci.com/gh/nipype/pydra)
[![codecov](https://codecov.io/gh/nipype/pydra/branch/master/graph/badge.svg)](https://codecov.io/gh/nipype/pydra)

The goal of pydra is to provide a lightweight Python dataflow engine for DAG construction, manipulation, and distributed execution.

Feature list:
1. Python 3.7+ using type annotation and [attrs](https://www.attrs.org/en/stable/)
2. Composable dataflows with simple node semantics. A dataflow can be a node of another dataflow.
3. `splitter` and `combiner` provides many ways of compressing complex loop semantics
4. Cached execution with support for a global cache across dataflows and users
5. Distributed execution, presently via ConcurrentFutures, SLURM,
and Dask (this is an experimental implementation with limited testing)

[[API Documentation](https://nipype.github.io/pydra/)]
[[PyCon 2020 Poster](https://docs.google.com/presentation/d/10tS2I34rS0G9qz6v29qVd77OUydjP_FdBklrgAGmYSw/edit?usp=sharing)]

### Tutorial
Pydra Tutorial can be find in [pydra-tutorial repository](https://github.com/nipype/pydra-tutorial).

The tutorial can be run locally or using Binder service: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nipype/pydra-tutorial/master?filepath=notebooks)

Please note that mybinder times out after an hour.

### Installation

```
pip install pydra
```

### Developer installation

Pydra requires Python 3.7+. To install in developer mode:

```
git clone git@github.com:nipype/pydra.git
cd pydra
pip install -e .[dev]
```

If you want to test execution with Dask:
```
git clone git@github.com:nipype/pydra.git
cd pydra
pip install -e .[dask]
```


It is also useful to install pre-commit:

```
pip install pre-commit
pre-commit
```
