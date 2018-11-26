Getting Started
===============

Install
-------

To install simply clone from the GitHub repository::

    cd $INSTALL_LOCATION
    git clone https://github.com/xiaoh/vt_dainv.git

Replace ``$INSTALL_LOCATION`` with the path to where you want to install the code.

When using the code you will need to source the following file in your terminal::

    source $INSTALL_LOCATION/vt_dainv/source/init_da

Prerequisites\:

* Python 2.7 or 3.x \*
* Python packages

    * NumPy
    * SciPy
    * Matplotlib

\* *2.7 required if using FOAM_Tau_Solver*

Developers
----------
Prerequisites\:

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
The source code is in ``docs/sphinx/``.
To compile run ``devtools/build_documentation``.
Sometimes it might be required to ``make clean`` from within ``docs/sphinx/``.

FOAM_Tau_Solver users
---------------------
If running FOAM_TAU_Solver you will need the following additional prerequisites\:

* OpenFOAM 2.x

Recommended\:

* ParaView
* ImageMagick

Also note that you will need to run the code with Python 2.7.
