# Copyright 2020 Virginia Polytechnic Institute and State University.
""" Inverse modeling main function. """

# standard library imports
import importlib

# third party imports
import numpy as np
import yaml

# local imports
# user-specified inverse method imported later with importlib
# user-specified physics model imported later with importlib


def run(model_file, inverse_method, nsamples,
        max_iterations=1, t_end=1., t_interval=1., save=True, rand_seed=None,
        inputs_model={}, inputs_inverse={}):
    """ Run DAFI.

    """ # TODO: docstring

    # random seed: do this before importing local modules that use np.random
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # initialize physics model
    spec = importlib.util.spec_from_file_location("model", model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    Model = getattr(model_module, 'Model')
    model = Model(
        nsamples, t_interval, t_end, max_iterations, inputs_model)

    # initialize inverse method
    Inverse = getattr(
        importlib.import_module('dafi.inverse'), inverse_method)
    inverse = Inverse(
        nsamples, t_interval, t_end, max_iterations, model, inputs_inverse)

    # solve the inverse problem
    print("Solving the inverse problem:" +
          f"\n  Physics Model:  {model.name}" +
          f"\n  Inverse Method: {inverse.name}")
    out = inverse.solve()

    # save & cleanup
    if save:
        inverse.save()
    try:
        inverse.clean()
    except AttributeError:
        pass

    return out
