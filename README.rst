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

pyanthem was primarily developed to interpret matrix-decomposed functional imaging datasets - for example, a dataset **V** with shape :code:`[height,width,time]`, is decomposed into two matrixes: **W** with shape :code:`[height*width,n]`, and **H** with shape :code:`[n,time]` such that :code:`H x W = V`. Here, n represents the number of variables represented by the decomposition. There are various techniques used to decompose matrixes - two popular techniques include Non-negative Matrix Factorization (NMF), and Singular Value Decomposition (SVD).

If you would prefer to keep things simple, you can skip matrix decomposition altogether and focus solely on converting raw data to audio - only working with the **H** matrix, where each row represents a variable and each column represents a time point.

If this is too much information to digest, don't worry! Try the example below to get more familiar with what the data looks like, and how pyanthem transforms it.

Installation
============

Easy Installation Using Anaconda
--------------------------------
First, make sure you have Anaconda_ (or Miniconda) installed before proceeding.

Create an environment and install the required packages:
   
.. code-block:: bash

   conda create -n pyanthem python=3.7 pip ffmpeg fluidsynth --channel conda-forge --channel nicthib

Next, activate the environment:
   
.. code-block:: bash

   conda activate pyanthem

Finally, install the pyanthem Python package using pip:
   
.. code-block:: bash

   pip install pyanthem

Manual Installation
-------------------

First, install ffmpeg and FluidSynth:

FFmpeg_:
   ffmpeg enables video creation and merging.

FluidSynth_:
   FluidSynth enables conversion of MIDI files to crisp, high quality sound files.
   
Finally, install pyanthem using pip: 

.. code-block:: bash

   pip install pyanthem

.. _`Python 3.7`: https://www.python.org/downloads/release/python-378/
.. _FFmpeg: https://ffmpeg.org/
.. _FluidSynth: https://github.com/FluidSynth/fluidsynth/wiki/Download
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda: https://www.anaconda.com/products/individual

*Note: Currently, pyanthem is tested to work on Python 3.7. Please use this version to minimize headaches.*

Optional: Using pyanthem in a Jupyter Notebook
-----------------------------------------------

To access the pyanthem environment in a Jupyter notebook, first install ipykernel in your environment:

.. code-block:: bash

   conda install -c anaconda ipykernel

After this, create the kernel:

.. code-block:: bash

   python -m ipykernel install --user --name=pyanthem

Once in a notebook, switch to the pyanthem kernel by selecting :code:`Kernel > Change kernel > pyanthem`

*Note: While the pyanthem kernel will now be available in any Jupyter notebook session, pyanthem will not function properly unless the Jupyter notebook is launched inside the pyanthem environment*.

Usage
=====

Before getting started, download some example datasets here_. For this, we are using the file called 'demo1.mat'

.. _here: https://github.com/nicthib/pyanthem/tree/master/datasets

Using pyanthem in GUI mode
--------------------------

First, import pyanthem and begin a pyanthem session:

.. code-block:: python
   
   import pyanthem
   pyanthem.run()

*Note: You may run into an error here where some packages are missing. Simply install them using pip, and try running pyanthem again.*

The first time you run pyanthem, it will download a necessary soundfont file - this will take a minute or two.

.. code-block::
   
   ♫ Initializing soundfont library...
   ♫ Downloading 17QuXRbApe0JTlYfBs7iSMCMu3xRWMHOV into...
   ♫ 238.3 MiB Done.

Once completed, the pyanthem GUI will initialize:

.. image:: https://github.com/nicthib/pyanthem/blob/media/GUI1.png

Next, load a dataset by clicking :code:`File > Load data...`. For this section, we will load the dataset :code:`demo1.mat`. Currently, you can import any .mat or hdf5 file that contains the following variables:

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
   g = pyanthem.run(display=False)

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

Decomposing raw datasets
------------------------

This feature is still a work in progress - results may vary for your specific dataset!

If you have some of your own data you would like to decompose into components for audiovisualization, you can utilize the CLI command process_raw() to accomplish this.

Example usage:

.. code-block:: python
   
   import pyanthem
   g = pyanthem.run(display=False)
   g.process_raw(file_in=r'path/to/your/file.mat',n_clusters=20,save=True)

Here, we first begin a CLI session using the display=False flag. Then, we load a .mat file for decomposition, clustering it into 20 components, and then create a decomposition using these clusters. The output - temporal and spatial components, are assigned to the workspace for further processing, and are also saved as a new file where the dataset was loaded from. Save is disabled by default, so make sure to set the save flag to True if you want to save the processed data.

*Note: You can also decompose a raw dataset in GUI mode using the :code:`File --> Load raw...` menu command.

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
