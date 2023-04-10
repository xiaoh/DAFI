
"""
Collection of gradient descent based optimization algorithms.
"""
import os

import numpy as np


# Main class, handles restart, saving, optimizing
class OptimizationAlgorithm(object):

    def __init__(self, alg_default_parameters, alg_restart_variables, objective,
                 restart=None, x=None, parameters=None, save=True,
                 save_directory='.'):
        # set parameters
        if parameters==None:
            parameters = {}
        self.parameters = parameters
        for param in alg_default_parameters.keys():
            if param not in self.parameters.keys():
                self.parameters[param] = alg_default_parameters[param]

        # whether to save iteration info, needed for restart
        self.save = save
        self.save_directory = save_directory

        # intermediate variables, needed for restart
        self.restart_variables = alg_restart_variables
        if restart is None:
            self.x = x
            self.iteration = 0
        elif restart == 'pretrain':
            self.iteration = 0
            file = os.path.join(self.save_directory, f'w.{self.iteration}')
            self.x = np.loadtxt(file)
        else:
            self.iteration = restart
            file = os.path.join(self.save_directory, f'w.{self.iteration}')
            self.x = np.loadtxt(file)
            for name in self.restart_variables.keys():
                file = os.path.join(self.save_directory, f'{name}.{self.iteration}')
                self.restart_variables[name] = np.loadtxt(file)

        self.objective = objective


    def optimize(self, n_iter):
        for i in range(n_iter+1):
            # self.J, grad, con_flag = self.objective(self.x)
            self.J, grad = self.objective(self.x)

            if self.save:
                self._save()
            # if con_flag: break
            self.iteration += 1
            self.x += self._step(grad)


    def _save(self, ):
        # weights
        file = os.path.join(self.save_directory, f'w.{self.iteration}')
        value = np.atleast_1d(self.x)
        np.savetxt(file, value)
        # restart variables
        for name, value in self.restart_variables.items():
            file = os.path.join(self.save_directory, f'{name}.{self.iteration}')
            value = np.atleast_1d(value)
            np.savetxt(file, value)


    def _step(self, grad):
        """ Implemented by specific algorithm. Returns delta_x.
        Updates restart_variables.
        """
        pass


# Child classes: algorithm specific update
class GradientDescent(OptimizationAlgorithm):

    def __init__(self, **kwargs):
        default_parameters = {
            'learning_rate': 1.0
            }
        restart_variables = {}
        super(self.__class__, self).__init__(
            alg_default_parameters = default_parameters,
            alg_restart_variables = restart_variables,
            **kwargs)

    def _step(self, grad):
        grad = np.squeeze(grad)
        return -grad * self.parameters['learning_rate']


class Momentum(OptimizationAlgorithm):

    def __init__(self, **kwargs):
        default_parameters = {
            'learning_rate': 1.0,
            'beta': 0.9,
            }
        restart_variables = {
            'V': 0.0,
            }
        super(self.__class__, self).__init__(
            alg_default_parameters = default_parameters,
            alg_restart_variables = restart_variables,
            **kwargs)

    def _step(self, grad):
        grad = np.squeeze(grad)
        self.restart_variables['V'] = \
            self.parameters['beta'] * self.restart_variables['V'] + \
            (1 - self.parameters['beta']) * grad
        return -self.parameters['learning_rate'] * self.restart_variables['V']


class RMSProp(OptimizationAlgorithm):

    def __init__(self, **kwargs):
        default_parameters = {
            'learning_rate': 0.001,
            'beta': 0.9,
            'eps': 1e-6,
            }
        restart_variables = {
            'S': 0.0,
            }
        super(self.__class__, self).__init__(
            alg_default_parameters = default_parameters,
            alg_restart_variables = restart_variables,
            **kwargs)

    def _step(self, grad):
        grad = np.squeeze(grad)
        self.restart_variables['S'] = \
            self.parameters['beta'] * self.restart_variables['S'] + \
            (1 - self.parameters['beta']) * grad**2
        step = \
            -self.parameters['learning_rate'] * grad / \
            np.sqrt(S_hat + self.parameters['eps'])
        return step


class Adam(OptimizationAlgorithm):

    def __init__(self, **kwargs):
        default_parameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            }
        restart_variables = {
            'V': 0.0,
            'S': 0.0,
            }
        super(self.__class__, self).__init__(
            alg_default_parameters = default_parameters,
            alg_restart_variables = restart_variables,
            **kwargs)

    def _step(self, grad):
        self.parameters['learning_rate'] = self.parameters['learning_rate'] #*0.99
        print('learning_rate = ', self.parameters['learning_rate'])
        grad = np.squeeze(grad)
        self.restart_variables['V'] = \
            self.parameters['beta1'] * self.restart_variables['V'] + \
            (1 - self.parameters['beta1']) * grad
        self.restart_variables['S'] = \
            self.parameters['beta2'] * self.restart_variables['S'] + \
            (1 - self.parameters['beta2']) * grad**2
        V_hat = \
            self.restart_variables['V'] / \
            (1 - self.parameters['beta1']**(self.iteration))
        S_hat = \
            self.restart_variables['S'] / \
            (1 - self.parameters['beta1']**(self.iteration))
        step = \
            -self.parameters['learning_rate'] * V_hat / \
            (np.sqrt(S_hat) + self.parameters['eps'])
        return step


class AMSGrad(OptimizationAlgorithm):

    def __init__(self, **kwargs):
        default_parameters = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-7,
            }
        restart_variables = {
            'V': 0.0,
            'S': 0.0,
            'S_hat': 0.0,
            }
        super(self.__class__, self).__init__(
            alg_default_parameters = default_parameters,
            alg_restart_variables = restart_variables,
            **kwargs)

    def _step(self, grad):
        grad = np.squeeze(grad)
        self.restart_variables['V'] = \
            self.parameters['beta1'] * self.restart_variables['V'] + \
            (1 - self.parameters['beta1']) * grad
        self.restart_variables['S'] = \
            self.parameters['beta2'] * self.restart_variables['S'] + \
            (1 - self.parameters['beta2']) * grad**2
        self.restart_variables['S_hat'] = \
            np.maximum(self.restart_variables['S'], self.restart_variables['S_hat'])
        step = \
            -self.parameters['learning_rate'] * self.restart_variables['V'] / \
            (np.sqrt(S_hat) + self.parameters['eps'])
        return step


# simple implementations (functions not classes), no restart or saving
def gradient_descent(func, x, n_iter, learning_rate):
    for _ in range(n_iter+1):
        _, grad = func(x)
        x -= grad * learning_rate
    return x


def momentum(func, x, n_iter, learning_rate, beta=0.9):
    V = 0.
    for _ in range(n_iter+1):
        _, grad = func(x)
        V  = beta * V + (1 - beta) * grad
        x -= V * learning_rate
    return x


def rmsprop(func, x, n_iter, learning_rate=0.001, beta=0.9, eps=1e-6):
    S = 0.
    for _ in range(n_iter+1):
        _, grad = func(x)
        S = beta * S + (1 - beta) * grad**2
        x -= grad * learning_rate / np.sqrt(S + eps)
    return x


def adam(func, x, n_iter, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    V = 0.
    S = 0.
    for i in range(n_iter+1):
        _, grad = func(x)
        V = beta1 * V + (1 - beta1) * grad
        S = beta2 * S + (1 - beta2) * grad**2
        V_hat = V / (1 - beta1**(i+1))
        S_hat = S / (1 - beta2**(i+1))
        x -= learning_rate * V_hat / (np.sqrt(S_hat) + eps)
    return x


def amsgrad(func, x, n_iter, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
    V = 0.
    S = 0.
    S_hat = 0.
    for i in range(n_iter+1):
        _, grad = func(x)
        V = beta1 * V + (1 - beta1) * grad
        S = beta2 * S + (1 - beta2) * grad**2
        S_hat = np.maximum(S, S_hat)
        x -= learning_rate * V / (np.sqrt(S_hat) + eps)
    return x
