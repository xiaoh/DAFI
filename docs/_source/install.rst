Getting Started
===============

Installation
------------

Using ``pip``::

    pip install dafi

From Github Repository::

    cd $INSTALL_LOCATION
    git clone https://github.com/xiaoh/dafi.git

Replace ``$INSTALL_LOCATION`` with the path to where you want to install the code.

When using the code you will need to add the following to your ``PATH`` and ``PYTHONPATH``. You can either run these commands in each new terminal or add it to your ``~/.bashrc`` file.::

    export PATH="$INSTALL_LOCATION/DAFI/bin:$PATH"
    export PYTHONPATH="$INSTALL_LOCATION/DAFI:$PYTHONPATH"

Prerequisites\:

* Python 3.8 
* Python packages

    * NumPy
    * SciPy
    * Matplotlib
    * PyYAML


Developers
----------
Prerequisites for building the documentation\:

* Sphinx
* Sphinx packages

    * RTD theme (read the docs)
    * bibtex extension

In order to use some of our scripts (like a pre-commit hook), we also recommend you install the following\:

* autopep8
* colordiff

See several useful scripts in the ``devtools`` directory.

Some notes for developers\:

* Follow pep8 and make sure that at a minimum your code passes ``autopep8``.
* All top level functions and public methods need a docstring. Follow the NumPy syntax.
* Recompile the documentation before committing if any docstrings have changed.
* If committing something that is incomplete or needs work always mark it with a ``#TODO: message`` comment.
* Try to keep variable names and style consistent with the rest of the code.
* Always update the documentation immediately if your changes warrant it.

Updating the documentation\:
The source code is in ``docs/_source/``.
To compile run ``make html`` from DAFI/docs. 
Sometimes it might be required to ``make clean``. 
Compiling the documentation locally is useful for developing, but the compiled html should not be pushed to GitHub. 
ReadTheDocs compiles it from source files. 

OpenFOAM
--------
For the OpenFOAM tutorial you will need to `install OpenFOAM <https://www.openfoam.com/download>`_ and `Paraview <https://www.paraview.org/download>`_ (recommended). 
We also find `ImageMagick <https://imagemagick.org/script/download.php>' and `Gnuplot <http://www.gnuplot.info/>` useful. 
