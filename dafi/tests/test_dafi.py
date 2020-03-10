#!/usr/bin/env python3
# Copyright 2020 Virginia Polytechnic Institute and State University.

# standard library imports
import unittest
import os
import shutil
import subprocess

# third party imports
import numpy as np

# local imports
import dafi


TUTORIALS_DIR = os.path.join('tutorials')


class TestEnKF(unittest.TestCase):
    """ Test the EnKF filter using the Lorentz model. """

    def setUp(self):
        self.dir = 'case'
        self.input_file = 'dafi.in'
        self.results_dir = 'results'
        model_file = 'model.py'
        model_input_file = 'model.in'

        # copy physics model file
        src = os.path.join(TUTORIALS_DIR, 'uq', 'uq.py')
        dst = os.path.join(self.dir, model_file)
        try:
            os.makedirs(self.dir)
            shutil.copyfile(src, dst)
        except:
            self.tearDown()
            os.makedirs(self.dir)
            shutil.copyfile(src, dst)

        # create input files

        def _create_dafi_input(filename):
            content = 'dafi:' + \
                f'\n    model_file: {model_file}' + \
                '\n    inverse_method: EnKF' + \
                '\n    nsamples: 5' + \
                '\n    max_iterations: 100' + \
                '\n    rand_seed: 1' + \
                '\n    verbosity: -1' + \
                '\n    convergence_option: discrepancy' + \
                '\n    convergence_factor: 1.2' + \
                f'\n    save_dir: {self.results_dir}' + \
                '\n    analysis_to_obs: True' + \
                '\n' + \
                '\ninverse:' + \
                '\n' + \
                '\nmodel:' + \
                f'\n    input_file: {model_input_file}'
            with open(filename, "w") as file:
                file.write(content)

        def _create_model_input(filename):
            content = 'x_init_mean: [0.5, 0.5]' + \
                '\nx_init_std: [0.1, 0.1]' + \
                '\nobs: [0.8, 2.]' + \
                '\nobs_std: [0.05, 0.05]'
            with open(filename, "w") as file:
                file.write(content)

        _create_dafi_input(os.path.join(self.dir, self.input_file))
        _create_model_input(os.path.join(self.dir, model_input_file))

        # run dafi
        bash_command = f"cd {self.dir}; dafi {self.input_file}"
        subprocess.call(bash_command, shell=True)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_HXa(self):
        HXa_org = np.array([
            [8.463987890948493353e-01, 8.576083701137400261e-01,
             8.551674246364239229e-01, 8.626731583860481889e-01,
             8.847827597673484368e-01]
            [1.947646865711072817e+00, 1.955315756794783866e+00,
             2.028639043768901029e+00, 2.021990441368010760e+00,
             2.015815695249300710e+00]
        ])
        file = os.path.join(self.dir, self.results_dir, 'HXa', 'HXa_0')
        HXa = np.loadtxt(file)
        check = np.allclose(HXa, HXa_org)
        self.assertEqual(check, True)

    def test_HXa(self):
        HXa_org = np.array([
            [8.463987890948493353e-01, 8.576083701137400261e-01,
             8.551674246364239229e-01, 8.626731583860481889e-01,
             8.847827597673484368e-01],
            [1.032670381430427442e+00, 1.031562459331395676e+00,
             1.054769419935357888e+00, 1.050511401065882700e+00,
             1.041897710974498015e+00]
        ])
        file = os.path.join(self.dir, self.results_dir, 'xa', 'xa_0')
        HXa = np.loadtxt(file)
        check = np.allclose(HXa, HXa_org)
        self.assertEqual(check, True)


if __name__ == '__main__':
    unittest.main()
