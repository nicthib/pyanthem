***********************************************************************
pyanthem: an audiovisualization tool to make your data more interesting
***********************************************************************

pyanthem is a Python_ tool that transforms three-dimensional time-varying datasets into a colorful, audible format. pyanthem boasts a variety of features: 

1) Raw data decomposition
2) Video and audio preview
3) A broad variety of video and audio parameters
4) Command-line reproduction via config files

Requirements
============
Python 3:
   Currently, pyanthem is tested to work on Python_ 3.7+. This will be 
   updated as more versions are tested.

pip:
   pip is needed for the installation of the pyanthem module and its
   dependencies.  Most python versions will have pip installed already, 
   see the  `pip installation`_ page for instructions if you do not 
   have pip.

ffmpeg:
   ffmpeg_ enables video creation and merging.

fluidsynth (optional, but **highly recommended**)
   fluidsynth_ is a powerful software synthesizer, which enables 
   conversion of data to crisp, high quality sound files.

git (optional):
  git_ allows pyanthem to download external audio files quickly and 
  easily.
  
.. _Python: https://www.python.org/
.. _`pip installation`: https://pip.pypa.io/en/latest/installing/
.. _git: https://git-scm.com/
.. _ffmpeg: https://ffmpeg.org/
.. _fluidsynth: http://www.fluidsynth.org/

Installation
============
Note: If you do not have working versions of the above listed 
requirements, it is recommended that you use miniconda_ for a
straightforward installation process.

Using miniconda:
----------------

First, download the pyanthem.yaml config file for Mac/Linux_, or for Windows_. 
Create the environment by navigating to the pyanthem.yaml file's location, 
and then by running::

   conda env create -f pyanthem.yaml

Next, activate the environment::

   conda activate pyanthem

If using Windows, there is an additional step, since conda-forge does not 
have Windows binaries for fluidsynth. Download the binaries here_. After 
unzipping, place the unzipped folder wherever you prefer, and then add
the location of the fluidsynth-1.1.1\usr\bin folder to your PATH (see 
`this guide`_ for adding to the PATH). For example, I added the directory:
C:\Users\Nic\Desktop\fluidsynth-1.1.1\usr\bin to my PATH.
   
Using pip
-----------

   python -m pip install pyanthem

.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Mac/Linux: https://drive.google.com/file/d/1HSZyFuU_9WmGTSVoVc-DuJzMi76CMseA
.. _Windows: https://drive.google.com/file/d/1HSZyFuU_9WmGTSVoVc-DuJzMi76CMseA
.. _`this guide`: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/
Usage
=====


Contributing
============

Step 1
------
- **Option 1**
    - Fork this repo!

- **Option 2**
    - Clone this repo to your local machine using 
    
    ```python
    git clone https://github.com/nicthib/pyanthem.git
    ```

Step 2
------
- Improve this repo!

Step 3
------

- Create a new pull request using `<https://github.com/nicthib/pyanthem/compare/>`_

Team
----

.. |niclogo| image:: https://avatars1.githubusercontent.com/u/34455769?v=3&s=200

.. csv-table::
   :header: Nic Thibodeaux

   |niclogo|
    `<http://github.com/nicthib>`
FAQ
---

- **How do I do *specifically* so and so?**
    - No problem! Just do this.

Support
-------
- Twitter: `@nicthibs`_

.. _`@nicthibs`: http://twitter.com/nicthibs
