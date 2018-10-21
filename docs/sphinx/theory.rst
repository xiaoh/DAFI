.. _theory:

Theory
======

Inverse Problem 
---------------

Considering a time-independent system, it can be prescribed as the following model:

.. math::

   x= [\theta, \alpha] 

where :math:`x` is state vector, :math:`\theta` is the state varibles characterizing the system behaviours and :math:`\alpha` represents the state augmentation parameters for system design.

Then the stochastic formulation of state vector and observation can be constructed for the system by adding an addictive random term in deterministic formulation as following:

.. math::

    x=x_0+\eta 

.. math::

    y=y_o+\epsilon

 
where :math:`\eta` is the random perturbation on the initial state vector :math:`x_0`, :math:`y_o` is the available observation of the system. :math:`\epsilon` is random observation error.
With random state vector and observation, we can leverage the Bayes's Theorem to get a posterior distribution by combining the prior distribution and data distribution which is called **inverse problem**.

.. math::

    p(x \mid y)  \propto p(x)p(y \mid x)

Thus the problem can be solved with data assimilation by finding state vector to maximize a posterior(MAP). 

Data Assimilation
-----------------

Data assimilation method can be divided as Kalman filter and Variational method. However 

Data assimilation can be sorted by variational method :cite:`le1986variational` and Kalman filter :cite:`kalman1960new`.The variational method use the optimal control theory :cite:`berkovitz2013optimal` optimal to reduce the misfit between observation and model realization in observed space, while Kalman filter is originated directly from the Bayesian formulation. However, the  variational method need much efforts on the adjoint model and the Kalman filter is quite costly to estimatethe statics of the state vector especially for high-dimension problem. To address these issues, the trending appraoch is to introduce ensemble technique, and many ensmeble based methods have been proposed.

This toolbox is focused on ensemble-based data assimilation methods. 

Ensemble Kalman Filtering (EnKF)
--------------------------------

In EnKF, the prior statistics is estimated by ensemble Monte Carlo sampling and each sample in kth DA step can be defined as:

.. math:: x_k^{(j)} = [\theta_k^{(j)},\alpha_k^{(j)}] \qquad 1 \leq j \leq N_{en}
    :label: bayes

where :math:`N_{en}` is the ensemble size. 

In :eq:`bayes`, MAP is equivalent to the determination of a variance minimizing analysis. Accordingly, the matrix is updated as:
 
.. math::
    x_{k+1}^{(j)}=x_k^{(j)}+K(y^{(j)}-H(x_k^{(j)}))
    :label: stateupdate_EnKF
 
with gain matrix :math:`K=PH^T(HPH^T+C)^{-1}`, :math:`P = \frac{1}{N_{en}-1}X'X'^T`, :math:`X'=(x^{(1)}-x^{(e)},x^{(2)}-x^{(e)},\dot,x^{(N_{en})}-x^{(e)})`
where H is operator that map from the state space to observation space, C is observation covariance error.

The procedure of EnKF can be summarized as follows:

#. Give a first guessed or prior state vector :math:`x^{(e)}`, and prescribe the prior and observation statistics respectively;

#. Realize Nen initial samples :math:`\{x^{(j)}\}_{j=1}^N` around :math:`x^{(e)}`;

#. Map the state vector to observation space by solving the RANS equation based on each sample and obtain HX;

#. Obtain gain matrix K and update the prior distribution based on :eq:`stateupdate_EnKF`;

#. Return to step 3 until further minimization cannot be achieved.

For more information, please refer to :cite:`xiao2016`


References
----------
.. bibliography:: Theory.bib
   :style: unsrt
