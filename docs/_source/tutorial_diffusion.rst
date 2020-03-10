
Tutorial: Diffusion
===================
This tutorial provides the instructions to run the code for one dimension field inversion problem based on heat diffusion equation using the provided Tutorials (located in the ''$tutorials/diffusion'').

Problem Description
-------------------
The heat diffusion equation for a rod can be expressed as following:

.. math::

   - \frac{d}{dx}(\mu \frac{du}{dx}) = f(x) \qquad x \in [0, 5]

where :math:`\mu` are unknown thermal diffusivity to be infered varying with the location in the rod, :math:`u` is the temperature and :math:`f(x)` is external heat source term which in this case is simplfied as :math:`f(x) = sin( 0.4 \pi x)`.

To solve the diffusion equation, the central difference scheme is used. The finite difference method is applied to replace the partial derivation of diffusivity in equation.

.. math::

   - \frac{d}{dx}(\mu \frac{du}{dx}) = -\frac{d\mu}{dx} \frac{du}{dx}+\mu\frac{du^2}{d^2x}=f(x)

with

.. math::

   \frac{du_i}{dx} = \frac{u_{i+1}-u_{i-1}}{2\delta x}

.. math::

   \frac{du^2}{d^2x} = \frac{u_{i+1}-2u_i+u_{i-1}}{\Delta x^2}

The difference scheme can be expressed as below:

.. math::
   (\frac{1}{2\Delta x}\frac{\mu_{i+1}-\mu_{i}}{\Delta x} + \frac{\mu_i}{\Delta x^2})u_{i+1} + (-\frac{2\mu_i}{\Delta x^2})\mu_i+(-\frac{1}{2\Delta x}\frac{\mu_{i+1}-\mu_{i}}{\Delta x} + \frac{\mu_i}{\Delta x^2}) u_{i-1} = f(x_i)

The spatial interval is 0.1, and the length of rod is 5. Thus the dimension of state varibles is 50.

Field Representation
--------------------

Field inverse problem is of great interest in reality, however to infer a field based on sparse observation also increases the ill-poseness of the problem. The high dimensionalty comparing to the limited number of samples will lead to the ununiqueness of the solution.

Therefore it is necessary to reduce the dimensionality to indirectly characterize the field. To represent the field, the most widely used method is Karhunen-Loeve expansion which is commonly used to represent the stochatic process through a combination of a set of orthogonal functions. In this case, the method is leveraged to reduce the dimension of state varibles.

The field of diffusivity is reconstructed by a set of deterministic functions with corresponding random variables:

.. math::

   \mu(x) = \sum_{i=1}^m \omega_i \phi_i(x)

where the subscript is the :math:`i`th mode :math:`\omega_i`, is a random variable and :math:`\phi_i(x)` is the deterministic basis set.

The prior of :math:`\mu(x)` is regarded as zero-mean Gaussian random fields, where the covariance of two different location (kernel function) is described as:

.. math::

   K(x,x')=\sigma(x)\sigma(x')exp(-\frac{|x-x'|^2}{l^2})

The prior variance :math:`\sigma(x)` is a constant or a spatially varying field, can be self-defined as:

.. math::

   \sigma =
   \begin{cases}
   & \frac{8}{5}x + 1 \qquad  &x \in [0, 2.5] \\
   & -\frac{8}{5}x + 9 \qquad &x \in (2.5, 5]
   \end{cases}

The correction length scale :math:`l` is simplified as the rod length in this tutorial.

The orthogonal basis function :math:`\phi_i(x)` take the form :math:`\phi_i(x)=\sqrt{\hat{\lambda_i}}\hat{\phi_i}(x)`, where :math:`\hat{lambda_i}` and :math:`\hat{\phi_i}(x)` are the eigenvalues and eigenvectors, respectively, of the kernel :math: `K` computed from the Fredholm integral equation:

.. math::

   \int K(x,x')\hat{\phi(x')dx' = \hat{\lambda}\hat{\phi}(x)


Thus we can obtain the diffusivity field through infering the randomized value :math:\omega_i based on the observation on temperature.


Building the Dynamic Model
--------------------------
This is a stationary system, and thus only forward model is concerned. The python script for diffusion model is created in the "source/dyn_models/diffusion.py"


Diffusion Model Files
^^^^^^^^^^^^^^^^^^^^^

Below is an overview of the files required to run the data assimilation for diffusion model in DA-Inv. The required files are listed below.

==================   =============================  =============================
**File Type**        **File Name**                  **Directory**
Input File           ``dafi.in``                    ``/tutorials/diffusion``
Input File           ``diffusion.in``               ``/tutorials/diffusion``
Forward Model        ``diffusion.py``               ``/source/dyn_models/``
==================   =============================  =============================

__init__
^^^^^^^^
The ``init`` function is\:

.. code-block:: python

        # save the main input
        self.nsamples = nsamples
        self.da_interval = da_interval
        self.t_end = t_end
        self.max_da_iteration = max_da_iteration

        # read input file
        param_dict = read_input_data(model_input)
        self.x_rel_std = float(param_dict['x_rel_std'])
        self.x_abs_std = float(param_dict['x_abs_std'])
        self.std_coef = float(param_dict['std_coef'])
        self.obs_rel_std = float(param_dict['obs_rel_std'])
        self.obs_abs_std = float(param_dict['obs_abs_std'])
        self.nmodes = int(param_dict['nmodes'])
        self.sigma = float(param_dict['sigma'])
        mu_init = float(param_dict['mu_init'])

        # required attributes
        self.name = '1D heat diffusion Equation'
        self.space_interval = 0.1
        self.max_length = 5
        self.nstate_obs = 10

        # create spatial coordinate and save
        self.x_coor = np.arange(
            0, self.max_length+self.space_interval, self.space_interval)
        np.savetxt('x_coor.dat', self.x_coor)

        # initialize state vector
        self.omega_init = np.zeros((self.nmodes))
        self.init_state = np.zeros(self.x_coor.shape)
        self.augstate_init = np.concatenate((
            self.init_state, np.zeros(self.nmodes)))
        # dimension of state space
        self.nstate = len(self.init_state)
        # dimension of augmented state space
        self.nstate_aug = len(self.augstate_init)

        # create source term fx
        S = np.zeros(self.nstate)
        for i in range(self.nstate - 1):
            S[i] = math.sin(2*np.pi*self.x_coor[i]/5)
        S = np.mat(S).T
        self.fx = S.A

        # create modes for K-L expansion
        cov = np.zeros((self.nstate, self.nstate))
        for i in range(self.nstate):
            for j in range(self.nstate):
                cov[i][j] = self.sigma * self.sigma * math.exp(
                    -abs(self.x_coor[i]-self.x_coor[j])**2/self.max_length**2)
        eigVals, eigVecs = sp.linalg.eigsh(cov, k=self.nmodes)
        ascendingOrder = eigVals.argsort()
        descendingOrder = ascendingOrder[::-1]
        eigVals = eigVals[descendingOrder]
        eigVecs = eigVecs[:, descendingOrder]

        # calculate KL modes: eigVec * sqrt(eigVal)
        KL_mode_raw = np.zeros([self.nstate, self.nmodes])
        for i in np.arange(self.nmodes):
            KL_mode_raw[:, i] = eigVecs[:, i] * np.sqrt(eigVals[i])
        # normalize the KL modes
        self.KL_mode = np.zeros([self.nstate, self.nmodes])
        for i in range(self.nmodes):
            self.KL_mode[:, i] = KL_mode_raw[:, i] / \
                np.linalg.norm(KL_mode_raw[:, i])
        np.savetxt('KLmodes.dat', self.KL_mode)

        # project the baseline to KL basis
        log_mu_init = [np.log(mu_init)] * self.nstate
        for i in range(self.nmodes):
            self.omega_init[i] = np.trapz(
                log_mu_init * self.KL_mode[:, i], x=self.x_coor)
            self.omega_init[i] /= np.trapz(self.KL_mode[:, i]
                                           * self.KL_mode[:, i], x=self.x_coor)
        np.savetxt('omega_init.dat', self.omega_init)

The inputs to the ``__init__ method`` is same as for Lorenz tutorial. In this method, we read the input file and get the input file for the diffusion model.

The input file is created as ``diffusion.in``. In the input file, we define the required parameters\:

.. literalinclude:: ../../tutorials/diffusion/diffusion.in

In the input file, we specify the number of modes and the field variance to represent the field. Then we specify the first guessed constant diffusivity, and the relative and absolute standard deviation to construct a random field. Also, we define the relative and absolute standard deviation for the observation.

.. note::
    With more modes, we can represent the field more flexiblely. However if the field is a low order curve, the results will lead to worse when introducing too many modes.

synthetic observations
^^^^^^^^^^^^^^^^^^^^^^
We give a synthetic observation on the temperature as:

.. math::
   \mu = 0.5+ 0.02 x^2


Running the code
----------------

As in the lorenz tutorial, to run the code first we need source ``init_da`` file if it has not been done yet. Then we can run the code by following the instruction below.

Step 1 : Write main input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify self-defined parameters in the 'dafi.in' file. The 'dafi.in' file is provided in the diffusion tutorial directory and shown below.

.. literalinclude::../../../tutorials/diffusion/dafi.in


In this file, we need to specify the number of samples (nsamples) and the maximum data assimilation iteration(max_da_iteration). This case is stationary case, and thus the end time and the data assimilation interval is 1.

Step 2 : Write forward model input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify self-defined parameters in the 'diffusion.in' file. The 'diffusion.in' file is provided in the diffusion tutorial directory and shown below.

.. literalinclude::../../../tutorials/diffusion/diffusion.in


In this file, we mainly need to specify the number of modes (nmodes), the synthetic true value for each mode coefficent (true_omega) and the relative standard deviation for observation (obs_rel_std)

Step 3 : Execute
^^^^^^^^^^^^^^^^

To execute the data assimilation for diffusion system, move to the directory of '$source'

Then move to the diffusion tutorial directory($tutorials/diffusion), and type code::
   dafi.py dafi.in

or the user can also simply type './run.sh' to run the tutorial. The process information will be saved in 'log.enkf' file at diffusion tutorial directory.

Step 4 : Postprocessing
^^^^^^^^^^^^^^^^^^^^^^^

'diffusion_plot.py' (located in '$tutorials/diffusion') is the postprocessing file to plot the inferred and observed field.

To execute the postprocessing, type code::
   ./diffusion_plot.py

to plot figures as shown below. Users can modify the 'diffusion_plot.py' for their own post-processing


References
----------
.. bibliography:: tutorial.bib
   :style: unsrt
   :labelprefix: B
