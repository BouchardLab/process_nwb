.. process_nwb

============
Installation
============

process_nwb is available for Python 3 and available on PyPI.


.. code-block:: bash

    $ pip install process-nwb

The latest development version
of the code can be installed from https://github.com/BouchardLab/process_nwb

.. code-block:: bash

    # use ssh
    $ git clone git@github.com:BouchardLab/process_nwb.git
    # or use https
    $ git clone https://github.com/BouchardLab/process_nwb.git
    $ cd process_nwb

To install into an active conda environment

.. code-block:: bash

    $ conda env update --file environment.yml
    $ pip install -e .

and with pip

.. code-block:: bash

    $ pip install -e . -r requirements.txt

Requirements
------------

Runtime
^^^^^^^

PyUoI requires

  * numpy
  * scipy
  * pynwb
  * mkl_fft

to run.

Develop
^^^^^^^

To develop process_nwb you will additionally need

  * pytest
  * flake8

to run the tests and check formatting.

Docs
^^^^

To build the docs you will additionally need

  * sphinx
  * sphinx_rtd_theme
