
mfuMain
=======
Inverse modeling main executable.

Example:
    >>> mfu_main.py <input_file>

Required inputs:
    * **dyn_model** (``str``) -
        Name of dynamic model module/package.
    * **dyn_model_input** (``str``) -
        Path to input file for the dynamic model.
    * **da_filter** (``str``) -
        Name of filter from dainv.da_filtering module.
    * **t_end** (``float``) -
        Final time step.
    * **da_interval** (``float``) -
        Time interval to perform data assimilation.
    * **nsamples** (``int``) -
        Number of samples for ensemble.

Optional inputs:
    * **report_flag** (``bool``, ``False``) -
        Call the filter's report method.
    * **plot_flag** (``bool``, ``False``) -
        Call the filter's plot method.
    * **save_flag** (``bool``, ``False``) -
        Call the filter's save method.
    * **rand_seed_flag** (``bool``, ``False``) -
        Use fixed random seed, for debugging.
    * **rand_seed** (``float``, ``1``) -
        Seed for numpy.random.

Other inputs:
    * As required by the chosen filter.
