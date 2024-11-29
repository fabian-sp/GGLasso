Getting started
======================

.. _Github: https://github.com/fabian-sp/GGLasso

Installation
^^^^^^^^^^^^^^^^

The official repository can be found on `Github`_
``GGLasso`` is available over ``pip`` and ``conda``. For installation from PyPi, simply run 

.. code-block::

     pip install gglasso

For ``conda``, use

.. code-block::

     conda install -c conda-forge gglasso

Alternatively, you can clone from Github and install a developer version with the command

.. code-block::

     python -m pip install --editable .


To make sure that everything works properly you can run unit tests with

.. code-block::

     pytest tests/ -v

To import from ``GGLasso`` in Python, type for example

.. code-block:: python

     from gglasso.problem import glasso_problem 


