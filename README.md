DAFI - Data Assimilation and Field Inversion
============================================
**DAFI** (Data Assimilation and Field Inversion) is an open-source, ensemble-based framework for solving inverse problems such as data assimilation and field inversion. Built with flexibility and extensibility in mind, it uses derivative-free Bayesian methods (ensemble Kalman filters) to infer physical fields from sparse observations while providing uncertainty quantification. DAFI integrates seamlessly with OpenFOAM and supports a wide range of physics models through a simple, object-oriented interface.

Website: https://dafi.readthedocs.io

## History:
- DAFI was originally developed at Dr. Heng Xiao's group at Virginia Tech. 
- In December 2022, Dr. Xiao moved to University of Stuttgart to hold the [Chair of Data-Driven Fluid Dynamics (DDSim)](https://www.hengx.org/) The code will be continuously maintained and updated by DDSim and collaborators.

If you use DAFI, please cite: C. A. Michelén Ströfer, X-L. Zhang, H. Xiao. DAFI: An open-source framework for ensemble-based data assimilation and field inversion. *Communications in Computational Physics* 29, pp. 1583-1622, 2021. DOI: [10.4208/cicp.OA-2020-0178](https://doi.org/10.4208/cicp.OA-2020-0178). Also available at: [arxiv: 2012.02651](https://arxiv.org/abs/2012.02651).

## List of publications using DAFI:

- X.-L. Zhang,  H. Xiao, X. Luo, G. He. Combining Direct and Indirect Sparse Data for Learning Generalizable Turbulence Models. *Journal of Computational Physics*, 489, 112272, 2023. DOI: [10.1016/j.jcp.2023.112272](https://doi.org/10.1016/j.jcp.2023.112272)

- MI Zafar, X Zhou, CJ Roy, D Stelter, H Xiao. Data-driven turbulence modeling approach for cold-wall hypersonic boundary layers. arXiv preprint [arXiv:2406.17446](https://arxiv.org/abs/2406.17446)

- X.-L. Zhang, H Xiao, S Jee, G He. Physical interpretation of neural network-based nonlinear eddy viscosity models. *Aerospace Science and Technology* 142 (a), 108632. DOI: [10.1016/j.ast.2023.108632](https://doi.org/10.1016/j.ast.2023.108632)  

- X.-L. Zhang, H. Xiao, X. Luo, G. He. Ensemble Kalman method for learning turbulence models from indirect observation data. *Journal of Fluid Mechanics*, 949(A26), 2022. DOI: [10.1017/jfm.2022.744](https://doi.org/10.1017/jfm.2022.744)

- C. A. Michelén Ströfer, X-L. Zhang, H. Xiao, O. Coutier-Delgosha. Enforcing boundary conditions on physical fields in Bayesian inversion. *Computer Methods in Applied Mechanics and Engineering* 367, 113097, 2020. DOI: [10.1016/j.cma.2020.113097](https://doi.org/10.1016/j.cma.2020.113097). Also available at: [arxiv: 1911.06683](https://arxiv.org/abs/1911.06683).

- X.-L. Zhang, C. A. Michelén Ströfer, H. Xiao. Regularization of ensemble Kalman methods for inverse problems. *Journal of Computational Physics*, 416, 109517, 2020. DOI: [10.1016/j.jcp.2020.109517](https://doi.org/10.1016/j.jcp.2020.109517). Also available at: [arxiv: 1910.01292](https://arxiv.org/abs/1910.01292).

- X.-L. Zhang, H. Xiao, T. Gomez, O. Coutier-Delgosha. Evaluation of ensemble methods for quantifying uncertainties in steady-state CFD applications with small ensemble sizes. *Computers & Fluids*, 203, 104530, 2020. DOI: [10.1016/j.compfluid.2020.104530](https://doi.org/10.1016/j.compfluid.2020.104530). Also available at: [arxiv: 2004.05541](https://arxiv.org/abs/2004.05541).

- X.-L. Zhang, H. Xiao, G. He, S. Wang. Assimilation of disparate data for enhanced reconstruction of turbulent mean flows. *Computers & Fluids*, 224, 104962, 2021. DOI: [10.1016/j.compfluid.2021.104962](https://doi.org/10.1016/j.compfluid.2021.104962).

- X.-L. Zhang, H. Xiao, G. He. Assessment of Regularized Ensemble Kalman Method for Inversion of Turbulence Quantity Fields. *AIAA Journal*, In Press, 2021. DOI: [10.2514/1.J060976](https://doi.org/10.2514/1.J060976).

- X.-L. Zhang, H. Xiao, T. Wu, G. He. Acoustic Inversion for Uncertainty Reduction in Reynolds-Averaged Navier–Stokes-Based Jet Noise Prediction. *AIAA Journal*, In Press, 2021. DOI: [10.2514/1.J060876](https://doi.org/10.2514/1.J060876).

Contributors:
-------------
* Carlos A. Michelén Ströfer (main developer)
* Xinlei Zhang
* Jianxun Wang
* Rui Sun
* Jinlong Wu

Contact: Carlos A. Michelén Ströfer; Heng Xiao
