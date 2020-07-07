********************************************************************
pyanthem: Transforming your datasets into a colorful, audible format
********************************************************************

Features:
=========

1) Raw data decomposition
2) A variety of video and audio parameters
3) In-place video and audio merge
4) GUI and Command-line interface for maximum flexibility
5) Example datasets to get you started

Requirements
============
Python 3.7:
   Currently, pyanthem is tested to work on `Python 3.7`_. This will be updated as more versions are tested.

ffmpeg:
   ffmpeg_ enables video creation and merging.

fluidsynth
   fluidsynth_ is a powerful software synthesizer, which enables conversion of data to crisp, high quality sound files.
  
.. _`Python 3.7`: https://www.python.org/downloads/release/python-378/
.. _ffmpeg: https://ffmpeg.org/
.. _fluidsynth: http://www.fluidsynth.org/

Installation
============
Note: If you do not have working installations of the above listed requirements, it is strongly recommended that you use miniconda_/Anaconda_ for a straightforward installation process. If you do not have either, miniconda is preferred as it is a faster install and takes up much less space.

If you do have the above requirements installed, you can simply install pyanthem using pip: :code:`pip install pyanthem`

Using Miniconda/Anaconda:
-------------------------

Create an environment and install the required packages::
   
    conda create -n pyanthem python=3.7 pip ffmpeg fluidsynth --channel conda-forge --channel nicthib

Next, activate the environment::
   
   conda activate pyanthem

Finally, install the pyanthem Python package using pip::
   
   pip install pyanthem

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda: https://www.anaconda.com/products/individual

Downloading Example datasets
----------------------------

If you want to get familiar with the datasets that pyanthem uses, download this collection of `datasets/config files`_

.. _`datasets/config files`: https://github.com/nicthib/anthem_datasets/archive/master.zip

Using pyanthem in a Jupyter Notebook
------------------------------------

To access the pyanthem environment in a Jupyter notebook, first install ipykernel in your environment::
   
   conda install -c anaconda ipykernel

After this, create the kernel::
   
   python -m ipykernel install --user --name=pyanthem

Once in a notebook, switch to the pyanthem kernel by selecting :code:`Kernel > Change kernel > pyanthem`

Usage
=====

Under construction!
-------------------

Team
====

.. |niclogo| image:: https://avatars1.githubusercontent.com/u/34455769?v=3&s=200

.. csv-table::
   :align: center
   :header: Nic Thibodeaux

   |niclogo|
   `http://github.com/nicthib`

FAQ
===

Under construction!
-------------------

Support
=======

- Twitter: `@nicthibs`_

.. _`@nicthibs`: http://twitter.com/nicthibs
