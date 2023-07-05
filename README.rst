
My Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to ImageNet-1k classification problem.

Tl;dr: To run the benchmark, run the following commands:
.. code-block::

	$ benchopt run benchmark_imagenet -s adamw[max_epochs=90,lr=0.001,amp=True] -d imagenet[data_path=path/to/imagenet]

where the path to ImageNet is user specific and should contains two subfolders: train/ and val/ containing respectively the official training set of ImageNet and the official validation set. All the options for the solver and the datasets can be bound in the sources of the corresponding files, in the paramaters dictionnary of the classes Solver and DataSet.

For now, this only runs on single GPU.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/agonon/benchmark_imagenet
   $ benchopt run benchmark_imagenet


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/agonon/benchmark_imagenet/workflows/Tests/badge.svg
   :target: https://github.com/agonon/benchmark_imagenet/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
