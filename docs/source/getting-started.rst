Getting started
======================

.. _Github: https://github.com/fabian-sp/GGLasso

Installation
^^^^^^^^^^^^^^^^

``GGLasso`` is available on `Github`_. Clone the repository and make sure you have all requirements installed. 
Then move to the directory and run

.. code-block::

     python setup.py

The above code installs a package called ``gglasso`` in your Python environment. In case you want to edit the source code and use the ``gglasso`` package without re-installing, you can run instead

.. code-block::

     python setup.py clean --all develop clean --all

To make sure that everything works properly you can run unit tests in ``glasso/tests``, for example

.. code-block::

     pytest gglasso/ -v

To import from ``GGLasso`` in Python, type for example

.. code-block:: python

     from gglasso.problem import gglasso_problem 


