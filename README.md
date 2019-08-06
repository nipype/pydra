
# Pydra <img src="pydra_logo.png" width="50">

A simple dataflow engine with scalable semantics.

[![Build Status](https://travis-ci.org/nipype/pydra.svg?branch=master)](https://travis-ci.org/nipype/pydra)
[![codecov](https://codecov.io/gh/nipype/pydra/branch/master/graph/badge.svg)](https://codecov.io/gh/nipype/pydra)

The goal of pydra is to provide a lightweight Python dataflow engine for DAG construction, manipulation, and distributed execution.

Feature list:
1. Python 3.7+ using type annotation and dataclasses
2. Composable dataflows with simple node semantics. A dataflow can be a node of another dataflow.
3. `splitter` and `combiner` provides many ways of compressing complex loop semantics
4. Cached execution with support for a global cache across dataflows and users
5. Distributed execution (presently via ConcurrentFutures and SLURM, more on their way)

### Tutorial
This tutorial will walk you through the main concepts of Pydra!
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nipype/pydra/master?filepath=tutorial%2Fnotebooks)

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

It is also useful to install pre-commit:

```
pip install pre-commit
pre-commit
```
