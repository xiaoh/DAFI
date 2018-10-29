
Tutorial: Lorenz63
==================

This section provides the instructions to run the code for Lorenz system using the provided Tutorials (located in the ''$tutorials/lorenz'').

Lorenz System
-------------
The Lorenz system is a chaotic dynamic system controlled by three ordinary differential equations(ODEs).

.. math::

   \frac{dx}{dt} = \sigma (y - x)

   \frac{dy}{dt} = \rho x - y -xz

   \frac{dz}{dt} = xy - \beta z

where :math:`\sigma, \rho, \beta` are parameters.

The time-varyting trajectory for Lorenz system is shown below.

.. figure:: _static/Lorenz63.png
   :width: 400pt

Data Assimilation Method
------------------------

To infer the optimal parameters for Lorenz equations based on the observation is the inverse problem in this tutorial.

Data assimilation method is to solve the inverse problem,

Different data assimilation methods are embeded in the code (located in the "$source/da_inv/da_filtering.py"). The list of usable data assimilation methods is shown below:

#. Ensemble Kalman Filtering :cite:`iglesias2013ensemble`
#. Ensemble Randomized Maximal Likelihood :cite:`gu2007iterative`
#. Ensemble Kalman Filtering-Multi Data Assimilation :cite:`evensen2018analysis`

Dynamic Model
-------------

The dynamic model in this tutorial is Lorenz model. The code for Lorenz model is located in the "source/dyn_models/lorenz.py"

*Lorenz dynamic model*

.. literalinclude:: ../../source/dyn_models/lorenz.py
   :language: python

Lorenz Model Files
------------------

Below is an overview of the files required to run the data assimilation for Lorenz model in DA-Inv. The required files are listed below.

==================   =============================  =============================
**File Type**        **File Name**                  **Directory**
Input File           ``dainv.in``                   ``/tutorials/lorenz``
Input File           ``lorenz.in``                  ``/tutorials/lorenz``
Dynamic Model        ``lorenz.py``   		          ``/source/dyn_models/``
==================   =============================  =============================

Run Tutorial
------------

Step 1 : Write main input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify self-defined parameters in the 'dainv.in' file. The 'dainv.in' file is provided in the lorenz tutorial directory and shown below.

.. literalinclude::../../../tutorials/lorenz/dainv.in
   :language: python

.. Note::

		Mainly need to specify the ensemble samples (nsamples) and the data assimilation interval (da_interval)

Step 2 : Write dynamic model input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify self-defined parameters in the 'lorenz.in' file. The 'lorenz.in' file is provided in the lorenz tutorial directory and shown below.

.. literalinclude::../../../tutorials/lorenz/lorenz.in
   :language: python

.. Note::

		Mainly need to specify which augmented parameter will be perturbed, the initial value for the state varibles, the relative standard deviation for state varibles and observation

Step 3 : Execute
~~~~~~~~~~~~~~~~

To execute the data assimilation for lorenz system, move to the directory of '$source' and type '. init_da.sh' to export the PATH. Then move to the lorenz directory($tutorials/lorenz), type './run.sh' to start the data assimilation process. The process information will be saved in 'log.enkf' file at lorenz tutorial directory.

Step 4 : Postprocessing
~~~~~~~~~~~~~~~~~~~~~~~

'Lorenz_plot.py' (located in '$tutorials/lorenz') is the postprocessing file to plot the trajectory of state varible 'x' 'y' and 'z'.

To execute the postprocessing, type './plot.sh' to plot figures as shown below. Users can modify the 'lorenz_plot.py' for their own post-processing

.. figure:: _static/timeSeries_DA_x.png
   :width: 400pt

.. figure:: _static/timeSeries_DA_y.png
   :width: 400pt

.. figure:: _static/timeSeries_DA_z.png
   :width: 400pt

References
------------------------
.. bibliography:: tutorial.bib
   :style: unsrt
   :labelprefix: B
