|CI/CD| |codecov| |PyPI| |Docs|

|Pydralogo|

======================
Pydra: Dataflow Engine
======================

A simple dataflow engine with scalable semantics.

Pydra is a rewrite of the Nipype engine with mapping and joining as
first-class operations. It forms the core of the Nipype 2.0 ecosystem.

The goal of pydra is to provide a lightweight Python dataflow engine for DAG
construction, manipulation, and distributed execution.

Feature list:
=============
1. Python 3.11+ using type annotation and `attrs <https://www.attrs.org/en/stable/>`_
2. Composable dataflows with simple node semantics. A dataflow can be a node of another dataflow.
3. `splitter` and `combiner` provides many ways of compressing complex loop semantics
4. Cached execution with support for a global cache across dataflows and users
5. Distributed execution, presently via ConcurrentFutures, SLURM and SGE, with support for PS/IJ and Dask available via plugins

`API Documentation <https://nipype.github.io/pydra/>`_

Learn more about Pydra
======================

* `SciPy 2020 Proceedings <http://conference.scipy.org/proceedings/scipy2020/pydra.html>`_
* `PyCon 2020 Poster <https://docs.google.com/presentation/d/10tS2I34rS0G9qz6v29qVd77OUydjP_FdBklrgAGmYSw/edit?usp=sharing>`_
* `Explore Pydra interactively <https://github.com/nipype/pydra-tutorial>`_ (the tutorial can be also run using Binder service)

|Binder|

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :alt: Binder


Please note that mybinder times out after an hour.

Installation
============

Pydra can be installed from PyPI using pip, noting that you currently need to specify
the 1.0-alpha version due to a quirk of PyPI version sorting, otherwise you will end up
with the old 0.25 version.

::

   pip install -–upgrade pip
   pip install pydra>=1.0a


If you want to install plugins for psij or dask you can by installing the relevant
plugin packages

::

    pip install pydra-workers-psij
    pip install pydra-workers-dask


Task implementations for various toolkits and workflows are available in task plugins,
which can be installed similarly

::

   pip install pydra-tasks-mrtrix3
   pip install pydra-tasks-fsl


Developer installation
======================

Pydra requires Python 3.11+. To install in developer mode::

    git clone git@github.com:nipype/pydra.git
    cd pydra
    pip install -e ".[dev]"

In order to run pydra's test locally::

    pytest pydra

We use `tox <https://tox.wiki/>`_ to test versions and dependency sets.
For example, to test on the minimum and latest dependencies, run::

    tox -e py311-min -e py313-latest

It is also useful to install pre-commit:

::

    pip install pre-commit
    pre-commit


.. |Pydralogo| image:: https://raw.githubusercontent.com/nipype/pydra/main/docs/source/_static/logo/pydra_logo.jpg
   :width: 200px
   :alt: pydra logo

.. |CI/CD| image:: https://github.com/nipype/pydra/actions/workflows/ci-cd.yml/badge.svg
   :alt: CI/CD
   :target: https://github.com/nipype/pydra/actions/workflows/ci-cd.yml

.. |codecov| image:: https://codecov.io/gh/nipype/pydra/branch/main/graph/badge.svg
   :alt: codecov
   :target: https://codecov.io/gh/nipype/pydra

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pydra.svg
   :alt: Supported Python versions
   :target: https://pypi.python.org/pypi/pydra

.. |PyPI| image:: https://img.shields.io/badge/pypi-1.0alpha-orange
   :alt: PyPI
   :target: https://pypi.org/project/pydra/1.0a0/

.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :alt: Documentation Status
   :target: https://nipype.github.io/pydra
