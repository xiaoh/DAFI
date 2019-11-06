#!/usr/bin/env python

# standard library imports
import unittest
import os
import shutil
import subprocess

# third party imports
import numpy as np

# local imports
import dafi.utilities as utils


class TestUtilities(unittest.TestCase):

    def setUp(self):
        self.testdir = 'test'
        self.file = "test_input.in"
        str_test = 'a = 1'
        str_test += '\n    '
        str_test += '\n   bb 2'
        str_test += '\n   bbb 2'
        str_test += '\n       '
        str_test += '\nc          True # ignore this!'
        str_test += '\n # d = 4'
        str_test += '\n          e:hello   # a 2'
        str_test += '\nf : world! # end'
        str_test += '\ng : 1,  2, 3,4,   5 #,6, # 7'
        utils.create_dir(self.testdir)
        test_file = open(self.testdir + os.sep + self.file, "w")
        test_file.write(str_test)
        test_file.close()

    def tearDown(self):
        os.remove(self.testdir + os.sep + self.file)
        os.rmdir(self.testdir)

    def test_utils(self):
        pattern = 'bb 2'
        subst = 'b 2'
        utils.replace(self.testdir + os.sep + self.file, pattern, subst)
        input_dict = utils.read_input_data(self.testdir + os.sep + self.file)
        input_dict['c'] = utils.str2bool(input_dict['c'])
        input_dict['g'] = utils.extract_list_from_string(
            input_dict['g'], sep=',', type_func=int)
        input_dict_correct = {'a': '1',
                              'b': '2',
                              'bb': '2',
                              'c': True,
                              'e': 'hello',
                              'f': 'world!',
                              'g': [int(1), int(2), int(3), int(4), int(5)]
                              }
        self.assertEqual(input_dict, input_dict_correct)


class TestEnKF(unittest.TestCase):
    """ Test the EnKF filter using the Lorentz model. """

    def setUp(self):
        self.dir = './lorenz'
        try:
            os.makedirs(self.dir)
        except:
            self.tearDown()
            os.makedirs(self.dir)
        self.main_input_file = 'dafi.in'
        self.model_input_file = 'lorenz.in'
        self.obs_file = self.dir + os.sep + 'truth.dat'

        def _create_main_input(filename):
            main_str = ''
            main_str += '\ndyn_model lorenz'
            main_str += '\ndyn_model_input ./lorenz.in'
            main_str += '\nda_filter EnKF'
            main_str += '\nmax_da_iteration 1'
            main_str += '\nt_end 0.75'
            main_str += '\nda_t_interval 0.25'
            main_str += '\nnsamples 10'
            main_str += '\nsave_flag True'
            main_str += '\nsave_dir ./results'
            main_str += '\nverbosity 2'
            main_str += '\nrand_seed_flag True'
            main_str += '\nrand_seed 1'
            main_in = open(filename, "w")
            main_in.write(main_str)
            main_in.close()

        def _create_model_input(filename):
            lorenz_str = ''
            lorenz_str += '\ndt_interval 0.025'

            lorenz_str += '\nx_init_mean = 1.0'
            lorenz_str += '\ny_init_mean = 1.2'
            lorenz_str += '\nz_init_mean = 1.'
            lorenz_str += '\nrho_init_mean = 33.0'
            lorenz_str += '\nx_init_std = 0.4'
            lorenz_str += '\ny_init_std = 2.0'
            lorenz_str += '\nz_init_std = 1.4'
            lorenz_str += '\nrho_init_std = 4.0'
            lorenz_str += '\nbeta = 2.7'
            lorenz_str += '\nsigma = 10.0'
            lorenz_str += '\nx_obs_rel_std = 0.10'
            lorenz_str += '\nz_obs_rel_std = 0.10'
            lorenz_str += '\nx_obs_abs_std = 0.05'
            lorenz_str += '\nz_obs_abs_std = 0.05'
            lorenz_str += '\nx_true = 1.0'
            lorenz_str += '\ny_true = 1.2'
            lorenz_str += '\nz_true = 1.0'
            lorenz_str += '\nrho_true = 28.0'
            lorenz_str += '\nbeta_true = 2.66667'
            lorenz_str += '\nsigma_true = 10.0'
            model_in = open(filename, "w")
            model_in.write(lorenz_str)
            model_in.close()

        _create_main_input(self.dir + os.sep + self.main_input_file)
        _create_model_input(self.dir + os.sep + self.model_input_file)

        bash_command = "cd {};".format(self.dir) + \
            "dafi.py {} > log.dafi".format(self.main_input_file)
        subprocess.call(bash_command, shell=True)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_HX(self):
        HX3_org = np.array([
            [-8.155832264643523288e+00, -8.049005901962603460e+00,
             -8.236407106016276458e+00, -9.243812811783399752e+00,
             -7.597852020120390293e+00, -9.440741026608749920e+00,
             -8.228514984834482959e+00, -8.457549588827466991e+00,
             -7.998435666424732027e+00, -8.185830170421192875e+00],
            [2.412252126467078739e+01, 2.398917488269303178e+01,
             2.421561490000541284e+01, 2.826725660453062616e+01,
             2.294640549453301759e+01, 2.864053366601696027e+01,
             2.532921124327318907e+01, 2.592581380781219025e+01,
             2.378528105793530401e+01, 2.429249016053097421e+01]
        ])
        HX3 = np.loadtxt(self.dir + os.sep + 'results' + os.sep + 'HXf' +
                         os.sep + 'HXf_3')
        check_HX3 = np.allclose(HX3, HX3_org)
        self.assertEqual(check_HX3, True)

    def test_obs(self):
        obs3_org = np.array([
            [-7.767292532655212689e+00, -6.955160037959183583e+00,
             -7.153301271421571350e+00, -6.372596047121427354e+00,
             -6.318800795923224634e+00, -7.104231197830644717e+00,
             -7.538443490266284464e+00, -7.742552720227275032e+00,
             -6.919574482954614858e+00, -7.187763904066735599e+00],
            [1.847620898917036669e+01, 1.923762477854051767e+01,
             1.793352599713311690e+01, 2.052371741590193110e+01,
             1.827325370916204506e+01, 2.155834469058122238e+01,
             1.994488815504615076e+01, 2.031844591476012951e+01,
             1.700023334409209497e+01, 1.948481794464024475e+01]
        ])
        obs3 = np.loadtxt(self.dir + os.sep + 'results' + os.sep + 'obs' +
                          os.sep + 'obs_3')
        check_obs3 = np.allclose(obs3, obs3_org)
        self.assertEqual(check_obs3, True)

    def test_obs_error(self):
        obserr3_org = np.array([
            [6.002661587585750302e-01, 0.000000000000000000e+00],
            [0.000000000000000000e+00, 3.861990782818556678e+00]
        ])
        obserr3 = np.loadtxt(self.dir + os.sep + 'results' + os.sep + 'R' +
                             os.sep + 'R_3')
        check_obserr3 = np.allclose(obserr3, obserr3_org)
        self.assertEqual(check_obserr3, True)

    def test_state_forecast(self):
        xf3_org = np.array([
            [-8.155832264643523288e+00, -8.049005901962603460e+00,
             -8.236407106016276458e+00, -9.243812811783399752e+00,
             -7.597852020120390293e+00, -9.440741026608749920e+00,
             -8.228514984834482959e+00, -8.457549588827466991e+00,
             -7.998435666424732027e+00, -8.185830170421192875e+00],
            [-9.938917107399788620e+00, -9.978316287092146908e+00,
             -1.005384805441996399e+01, -1.111942613729705975e+01,
             -8.817653002135942941e+00, -1.134503645848653619e+01,
             -9.843449753924545576e+00, -1.001058283938535887e+01,
             -9.712477354545283603e+00, -9.761528892152032455e+00],
            [2.412252126467078739e+01, 2.398917488269303178e+01,
             2.421561490000541284e+01, 2.826725660453062616e+01,
             2.294640549453301759e+01, 2.864053366601696027e+01,
             2.532921124327318907e+01, 2.592581380781219025e+01,
             2.378528105793530401e+01, 2.429249016053097421e+01],
            [2.716407721730455904e+01, 2.746693063644474364e+01,
             2.721296853910794411e+01, 3.059908937716555499e+01,
             2.541762004437724087e+01, 3.077911525432359596e+01,
             2.813083377372663563e+01, 2.837554952837561473e+01,
             2.686288084351753724e+01, 2.693712172871166999e+01]
        ])
        xf3 = np.loadtxt(self.dir + os.sep + 'results' + os.sep + 'Xf' +
                         os.sep + 'Xf_3')
        check_xf3 = np.allclose(xf3, xf3_org)
        self.assertEqual(check_xf3, True)

    def test_state_analysis(self):
        xa3_org = np.array([
            [-7.449406711293732108e+00, -7.286132366890639034e+00,
             -7.307528838128065196e+00, -7.761530566489565075e+00,
             -6.802934334062279653e+00, -8.148622115174015690e+00,
             -7.484697298673212096e+00, -7.683723202627326110e+00,
             -7.015162736993404202e+00, -7.437814744740546935e+00],
            [-9.105128968423880309e+00, -9.064317365730168774e+00,
             -8.945971235203383998e+00, -9.328475377162465776e+00,
             -7.862456039544789554e+00, -9.787200314972549720e+00,
             -8.959866792934111146e+00, -9.091411199875981453e+00,
             -8.541087309751269174e+00, -8.866832083224410965e+00],
            [2.164960555114828367e+01, 2.136539322574414612e+01,
             2.100364336324395254e+01, 2.322088174910901515e+01,
             2.022206578072200500e+01, 2.423003191318250060e+01,
             2.274487555024362351e+01, 2.323701321012963206e+01,
             2.038052744448347653e+01, 2.171464836671729870e+01],
            [2.505274183543566480e+01, 2.521355395781200670e+01,
             2.445940779263752063e+01, 2.625022197224871334e+01,
             2.307509924014139457e+01, 2.698158833683224955e+01,
             2.591885500630560912e+01, 2.607421504235820109e+01,
             2.394539641907562455e+01, 2.472468732624123433e+01]

        ])
        xa3 = np.loadtxt(self.dir + os.sep + 'results' + os.sep + 'Xa' +
                         os.sep + 'Xa_3')
        check_xa3 = np.allclose(xa3, xa3_org)
        self.assertEqual(check_xa3, True)

    def test_misfit(self):
        misfit_org = np.array([
            3.272550393345796671e+00,
            1.225508994173762911e+00,
            9.657890853853755564e-01
        ])
        misfit = np.loadtxt(
            self.dir + os.sep + 'results' + os.sep + 'misfit_norm')
        check_misfit = np.allclose(misfit, misfit_org)
        self.assertEqual(check_misfit, True)

    def test_sigma_model(self):
        sigma_model_org = np.array([
            9.769347662985422787e+00,
            1.586135965498215006e+00,
            9.504807634431071683e-01
        ])
        sigma_model = np.loadtxt(
            self.dir + os.sep + 'results' + os.sep + 'sigma_HX')
        check_sigma_model = np.allclose(sigma_model, sigma_model_org)
        self.assertEqual(check_sigma_model, True)

    def test_sigma_obs(self):
        sigma_obs_org = np.array([
            9.468700346592401340e-01,
            1.441269310292525363e+00,
            1.056202743508216413e+00
        ])
        sigma_obs = np.loadtxt(
            self.dir + os.sep + 'results' + os.sep + 'sigma_obs')
        check_sigma_obs = np.allclose(sigma_obs, sigma_obs_org)
        self.assertEqual(check_sigma_obs, True)


if __name__ == '__main__':
    unittest.main()
