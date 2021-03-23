********************************************************************
pyanthem: Transforming your datasets into a colorful, audible format
********************************************************************

Features:
=========

1) Converts three-dimensional datasets to visually pleasing, audible representations
2) A variety of video and audio parameters
3) In-place video and audio merge
4) GUI and Command-line interface for maximum flexibility
5) Example datasets to get you started
6) Headache-free installation with an Anaconda environment

Usage
=====

pyanthem was primarily developed to interpret matrix-decomposed functional imaging datasets - for example, a dataset **V** with shape :code:`[height,width,time]`, is decomposed into two matrixes: **W** with shape :code:`[height*width,n]`, and **H** with shape :code:`[n,time]` such that :code:`H x W = V`. Here, n represents the number of variables represented by the decomposition. There are various techniques used to decompose matrixes - two popular techniques include Non-negative Matrix Factorization (NMF), and Singular Value Decomposition (SVD).

Here's a visual illustration of matrix decomposition - note that in this example, :code:`n=2`:

.. image:: https://upload.wikimedia.org/wikipedia/commons/f/f9/NMF.png

If you would prefer to keep things simple, you can skip matrix decomposition altogether and focus solely on converting raw data to audio - only working with the **H** matrix, where each row represents a variable and each column represents a time point.

If this is too much information to digest, don't worry! Try the example below to get more familiar with what the data looks like, and how pyanthem transforms it.

Requirements
============

*See installation guide below before proceeding!*

`Python 3.7`_:
   Currently, pyanthem is tested to work on Python 3.7. This will be updated as more versions are tested.

FFmpeg_:
   ffmpeg enables video creation and merging.

FluidSynth_:
   FluidSynth enables conversion of MIDI files to crisp, high quality sound files.
   
If you do have the above requirements installed, you can install pyanthem using pip: :code:`pip install pyanthem`

Conda (optional, but highly recommended):
   Conda enables simple and reliable package installation. Use Miniconda_ for a minimal installation, or Anaconda_ otherwise.

.. _`Python 3.7`: https://www.python.org/downloads/release/python-378/
.. _FFmpeg: https://ffmpeg.org/
.. _FluidSynth: http://www.fluidsynth.org/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda: https://www.anaconda.com/products/individual

Installation
============
Note: If you do not have working installations of the listed requirements (Python 3.7 + fluidsynth + ffmpeg), it is **strongly recommended** that you use Miniconda_/Anaconda_ for a straightforward installation process. If you do not have either, Miniconda is preferred as it is a faster install and takes up much less space than Anaconda.

Using Miniconda/Anaconda:
-------------------------

Create an environment and install the required packages::
   
   conda create -n pyanthem python=3.7 pip ffmpeg fluidsynth --channel conda-forge --channel nicthib

Next, activate the environment::
   
   conda activate pyanthem

Finally, install the pyanthem Python package using pip::
   
   pip install pyanthem

Example datasets and Scripts
----------------------------

If you want to get familiar with the datasets that pyanthem uses and try some applications of pyanthem, clone this repository or download `here.`

.. _`here`: https://github.com/nicthib/pyanthem/archive/master.zip

Using pyanthem in a Jupyter Notebook
-----------------------------------------------

To access the pyanthem environment in a Jupyter notebook, first install ipykernel in your environment::
   
   conda install -c anaconda ipykernel

After this, create the kernel::
   
   python -m ipykernel install --user --name=pyanthem

Once in a notebook, switch to the pyanthem kernel by selecting :code:`Kernel > Change kernel > pyanthem`

*Note: While the pyanthem kernel will now be available in any Jupyter notebook session, pyanthem will not function properly unless the Jupyter notebook is launched inside the pyanthem environment*.

Example
=======

Using pyanthem in GUI mode
--------------------------

First, import pyanthem and begin a pyanthem session:

.. code-block:: python
   
   import pyanthem
   pyanthem.run()

The first time you run pyanthem, it will download a necessary soundfont file - this will take a minute or two.

.. code-block::
   
   ♫ Initializing soundfont library...
   ♫ Downloading 17QuXRbApe0JTlYfBs7iSMCMu3xRWMHOV into...
   ♫ 238.3 MiB Done.

Once completed, the pyanthem GUI will initialize:

.. image:: https://github.com/nicthib/pyanthem/blob/media/GUI1.png

Next, load a dataset by clicking :code:`File > Load from .mat`. For this section, we will load the dataset :code:`demo1.mat`. Currently, you can import any .mat or hdf5 file that contains the following variables:

1) Temporal variable (**H, required**): A 2D matrix of shape :code:`[n,t]`, where each row is a component and each column is a time-point. This variable is referred to as **"H"** in the pyanthem environment.

2) Spatial variable (**W, optional**): A 3D matrix of shape :code:`[h,w,n]`, where h and w represent the spatial height and width of your dataset. If this variable is not given, no video output is possible.

3) Framerate (**fr, optional**): A single float value, representing the frame rate of your dataset in Hz. If a framerate is not given, pyanthem will provide a default.

*Note: Make sure to only include these variables in your file to avoid any errors. You can name them however you like, but make sure there are only one of each variable.* 

Once loading is complete, the GUI should update with default options, and plots of **H** and **W**:

.. image:: https://github.com/nicthib/pyanthem/blob/media/GUI2.png

The bottom left plots show two representations of the dataset: A preview of the output movie (left), and a visualization of what components are included and the colormap selection. The right two plots show raw representations of **H** (top), and a visualization of the audio output file (right). Lighter colors indicate loud notes, and darker colors indicate quiet notes, with black indicating silence.

From here, you can adjust parameters, preview the output, and finally save video and audio files. If you want to check how your parameter adjustments impact your audivisualization, click the **Update** button, and your changes will be reflected. Any issues with your selected parameters will be indicated in the white status box. Try adjusting a few parameters and observing how the plots change.

Finally, render output files with the :code:`Save --> Write A/V then merge` menu command.

Congratulations - you've created your first audiovisualization!

Using pyanthem in CLI (command-line interface) mode
---------------------------------------------------

pyanthem's CLI mode is useful for running batch conversions of large amounts of data once you are happy with your audiovisualization parameters, or creating more complex audiovisualizions that use multiple datasets and instruments. CLI mode is not recommended to use until you have used the GUI and are comfortable with the parameters and usage.

To run pyanthem in CLI mode, pass the argument :code:`display=False`, and assign the :code:`.run()` method to a variable:

.. code-block:: python
   
   import pyanthem
   g=pyanthem.run(display=False)

Next, load a dataset and config file using the :code:`.load_data()` and :code:`.load_config()` methods. You can pass an explicit file name to the :code:`file_in` argument, or pass none to recieve a file select prompt (note the use of the leading :code:`r` when naming a file location):

.. code-block:: python
   
   g.load_data(file_in=r'path/to/your/file.mat')
   g.load_config(file_in=r'path/to/your/config.p')

Finally, render the audio and video file, then merge the files using the :code:`.write_audio()`, :code:`.write_video()` and :code:`.merge()` methods:

.. code-block:: python
   
   g.write_audio()
   g.write_video()
   g.merge()

Once you're comfortable with this syntax, you can combine all of these steps into a single line, write a merged video with the :code:`.write_AV()` method, and even remove the intermediate files using the :code:`.cleanup()` method:

.. code-block:: python
   
   data_file = r'path/to/your/file.mat'
   config_file = r'path/to/your/config.p'
   g.load_data(file_in=data_file).load_config(file_in=config_file).write_AV().cleanup()

Congratulations - you've created your first audiovisualization in CLI mode!

Team
====

.. |niclogo| image:: https://avatars1.githubusercontent.com/u/34455769?v=3&s=200

+---------------------------+
| Nic Thibodeaux            |
+===========================+
| |niclogo|                 |
+---------------------------+
| http://github.com/nicthib |
+---------------------------+

FAQ
===

Under construction!
-------------------

Support
=======

- Twitter: `@nicthibs`_

.. _`@nicthibs`: http://twitter.com/nicthibs
