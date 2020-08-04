.. Pydra: A simple dataflow engine with scalable semantics documentation master file, created by
   sphinx-quickstart on Fri Jan  3 13:52:41 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pydra: A simple dataflow engine with scalable semantics's documentation!
===================================================================================

Pydra is a new lightweight dataflow engine written in Python.
Pydra is developed as an open-source project in the neuroimaging community,
but it is designed as a general-purpose dataflow engine to support any scientific domain.

Scientific workflows often require sophisticated analyses that encompass a large collection
of algorithms.
The algorithms, that were originally not necessarily designed to work together,
and were written by different authors.
Some may be written in Python, while others might require calling external programs.
It is a common practice to create semi-manual workflows that require the scientists
to handle the files and interact with partial results from algorithms and external tools.
This approach is conceptually simple and easy to implement, but the resulting workflow
is often time consuming, error-prone and difficult to share with others.
Consistency, reproducibility and scalability demand scientific workflows
to be organized into fully automated pipelines.
This was the motivation behind Pydra - a new dataflow engine written in Python.

The Pydra package is a part of the second generation of the Nipype_ ecosystem
--- an open-source framework that provides a uniform interface to existing neuroimaging
software and facilitates interaction between different software components.
The Nipype project was born in the neuroimaging community, and has been helping scientists
build workflows for a decade, providing a uniform interface to such neuroimaging packages
as FSL_, ANTs_, AFNI_, FreeSurfer_ and SPM_.
This flexibility has made it an ideal basis for popular preprocessing tools,
such as fMRIPrep_ and C-PAC_.
The second generation of Nipype ecosystem is meant to provide additional flexibility
and is being developed with reproducibility, ease of use, and scalability in mind.
Pydra itself is a standalone project and is designed as a general-purpose dataflow engine
to support any scientific domain.

The goal of Pydra is to provide a lightweight dataflow engine for computational graph construction,
manipulation, and distributed execution, as well as ensuring reproducibility of scientific pipelines.
In Pydra, a dataflow is represented as a directed acyclic graph, where each node represents a Python
function, execution of an external tool, or another reusable dataflow.
The combination of several key features makes Pydra a customizable and powerful dataflow engine:

- Composable dataflows: Any node of a dataflow graph can be another dataflow, allowing for nested
  dataflows of arbitrary depths and encouraging creating reusable dataflows.

- Flexible semantics for creating nested loops over input sets: Any Task or dataflow can be run
  over input parameter sets and the outputs can be recombined (similar concept to Map-Reduce_ model,
  but Pydra extends this to graphs with nested dataflows).

- A content-addressable global cache: Hash values are computed for each graph and each Task.
  This supports reusing of previously computed and stored dataflows and Tasks.

- Support for Python functions and external (shell) commands: Pydra can decorate and use existing
  functions in Python libraries alongside external command line tools, allowing easy integration
  of existing code and software.

- Native container execution support: Any dataflow or Task can be executed in an associated container
  (via Docker or Singularity) enabling greater consistency for reproducibility.

- Auditing and provenance tracking: Pydra provides a simple JSON-LD-based message passing mechanism
  to capture the dataflow execution activties as a provenance graph. These messages track inputs
  and outputs of each task in a dataflow, and the resources consumed by the task.

.. _Nipype: https://nipype.readthedocs.io/en/latest/
.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
.. _ANTs: http://stnava.github.io/ANTs/
.. _AFNI: https://afni.nimh.nih.gov/
.. _FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
.. _SPM: https://www.fil.ion.ucl.ac.uk/spm/
.. _fMRIPrep: https://fmriprep.org/en/stable/
.. _C-PAC: https://fcp-indi.github.io/docs/latest/index
.. _Map-Reduce: https://en.wikipedia.org/wiki/MapReduce

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   changes
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
