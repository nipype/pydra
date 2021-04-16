|GHAction| |CircleCI| |codecov|

|Pydralogo|

.. |Pydralogo| image:: https://raw.githubusercontent.com/nipype/pydra/master/docs/logo/pydra_logo.jpg
   :width: 200px
   :alt: pydra logo

.. |GHAction| image:: https://github.com/nipype/pydra/workflows/Pydra/badge.svg
   :alt: GitHub Actions CI
   :target: https://github.com/nipype/Pydra/actions

.. |CircleCI| image:: https://circleci.com/gh/nipype/pydra.svg?style=svg
   :alt: CircleCI

.. |codecov| image:: https://codecov.io/gh/nipype/pydra/branch/master/graph/badge.svg
   :alt: codecov

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
1. Python 3.7+ using type annotation and `attrs <https://www.attrs.org/en/stable/>`_
2. Composable dataflows with simple node semantics. A dataflow can be a node of another dataflow.
3. `splitter` and `combiner` provides many ways of compressing complex loop semantics
4. Cached execution with support for a global cache across dataflows and users
5. Distributed execution, presently via ConcurrentFutures, SLURM, and Dask (this is an experimental implementation with limited testing)

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

::

    pip install pydra


Note that installation fails with older versions of pip on Windows. Upgrade pip before installing:

::

   pip install –upgrade pip
   pip install pydra


Developer installation
======================

Pydra requires Python 3.7+. To install in developer mode:

::

    git clone git@github.com:nipype/pydra.git
    cd pydra
    pip install -e ".[dev]"


In order to run pydra's test locally:

::

    pytest -vs pydra


If you want to test execution with Dask:

::

    git clone git@github.com:nipype/pydra.git
    cd pydra
    pip install -e ".[dask]"



It is also useful to install pre-commit:

::

    pip install pre-commit
    pre-commit
