
dafi (executable)
=================
DAFI main executable.

Parses the input file and runs ``dafi.run()``.
Saves the DAFI version (Git commit hash) and the run time to hidden
files (*'.dafi_ver'* and *'.time'*, respectively).

**Example**

.. code-block:: python

    >>> dafi <input_file>

**Input File**
    The input file is written using yaml syntax.
    The input file contains the following three dictionaries:

    * **dafi** (required) - see :py:func:`dafi.main.run` for inputs.
    * **inverse** - see inputs in chosen inverse method in
      :py:mod:`dafi.inverse`. This corresponds to the *'inputs_inverse'*
      input to :py:func:`dafi.main.run`.
    * **model** - inputs as required by user-provided physics model.
      This corresponds to the *'inputs_model'* input to
      :py:func:`dafi.main.run`.
