"""
    Construct Python (training) and C++ (evaluation) neural networks.
    Functions to format/work with tensors (weights, Jacobian).
"""

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


# Neural Network
class NN(tf.keras.Model):
    def __init__(self, ninputs, noutputs, nhlayers, nnodes, alpha=0.0):
        super(NN, self).__init__()
        self._build_nn(ninputs, noutputs, nhlayers, nnodes, alpha)
        self.build((1, ninputs))
        self._initialize_weights(alpha)


    def _build_nn(self, ninputs, noutputs, nhlayers, nnodes, alpha=0.0):
        """ Deep neural network with constant size hidden layers and
        (leaky) ReLU activation.
        """
        self.layers_list = []
        initializer = tf.random_normal_initializer( # TODO: need fix initializer
            mean=0.0, stddev=1.0, seed=None)
        layer_properties = {"units": nnodes,
                            "activation": 'linear',
                            "use_bias": True,
                            "kernel_initializer": initializer,
                            # "kernel_initializer": tf.keras.initializers.HeNormal(),
                            "bias_initializer": 'zeros',
                            }
        for _ in range(nhlayers):
            hidden_layer = tf.keras.layers.Dense(**layer_properties)
            activation = tf.keras.layers.ReLU(negative_slope=alpha)
            self.layers_list.append(hidden_layer)
            self.layers_list.append(activation)
        layer_properties["units"] = noutputs
        final_layer = tf.keras.layers.Dense(**layer_properties)
        self.layers_list.append(final_layer)


    def _initialize_weights(self, alpha=0.0):
        """ Kaiming He initialization. """
        # weights: He initialization
        for weight in self.trainable_variables[::2][:-1]:
            fanin = weight.shape[0]
            var = 2 / ((1 + alpha**2) * fanin)
            weight.assign(weight.numpy() * np.sqrt(var))
        fanin = self.trainable_variables[::2][-1].shape[0]
        var = 1 / fanin
        self.trainable_variables[::2][-1].assign(
            self.trainable_variables[::2][-1].numpy() * np.sqrt(var))


    def set_weights(self, values):
        """ new_weights list of numpy arrays: [w1, b1, ..., wN, bN].
        N is the number of hidden layers + 1 (output layer).
        """
        for weight, value in zip(self.trainable_variables, values):
            weight.assign(value)


    def call(self, x):
        """ Evaluate network outputs for given input. """
        for layer in self.layers_list:
            x = layer(x)
        return x


# C++ model
class CppNorm(tf.keras.layers.Layer):
    """ Normalize layer. Used to normalize inpusts in cpp_build_nn. """
    def __init__(self, units, xmin=0.0, xmax=1.0):
        super(CppNorm, self).__init__()
        self.units = units
        self.xmin = xmin
        self.xmax = xmax

    # def build(self, input_shape):

    def call(self, inputs):
        return (inputs - self.xmin)/(self.xmax - self.xmin)


def cpp_build_nn(ninputs, noutputs, nhlayers, nnodes, scale_min, scale_max):
    """ Deep neural network with constant size hidden layers and
    ReLU activation.
    """
    # layers list
    layers_list = []
    # normalization layer
    layers_list.append(CppNorm(units=ninputs, xmin=scale_min, xmax=scale_max))
    initializer = tf.random_normal_initializer( # TODO: need fix initializer
        mean=0.0, stddev=1.0, seed=None)
    # hidden layers
    layer_properties = {"units": nnodes,
                        "activation": 'relu',
                        "use_bias": True,
                        "kernel_initializer": initializer,
                        # "kernel_initializer": tf.keras.initializers.HeNormal(),
                        "bias_initializer": 'zeros',
                        }
    for _ in range(nhlayers):
        layers_list.append(tf.keras.layers.Dense(**layer_properties))
    # final layer
    layer_properties["units"] = noutputs
    layer_properties["activation"] = 'linear'
    layers_list.append(tf.keras.layers.Dense(**layer_properties))
    # create model
    model = tf.keras.Sequential(layers_list)
    model.build((1, ninputs))
    model._set_inputs(tf.TensorSpec([None, ninputs], tf.float64, name='inputs'))
    return model


def cpp_set_weights(model, values):
        """ values: list of numpy arrays: [w1, b1, ..., wN, bN].
        N is the number of hidden layers + 1 (output layer).
        """
        for weight, value in zip(model.trainable_variables, values):
            weight.assign(value)
        return model



# formatting weight tensors
def weights_shape(w):
    """ List of tensors to list of shape tuples. """
    shapes = []
    for iw in w:
        shapes.append(iw.shape)
    return shapes


def shapes_to_sizes(shapes):
    sizes = []
    for shape in shapes:
        isize = 1
        for ishape in shape:
            isize *= ishape
        sizes.append(isize)
    return sizes


def reshape_weights(values_flat, shapes):
    """ From flat vector to list of arrays for TensorFlow.
    """
    w_reshaped = []
    sizes = shapes_to_sizes(shapes)
    i = 0
    for shape, size in zip(shapes, sizes):
        w_reshaped.append(values_flat[i:i+size].reshape(shape))
        i += size
    return w_reshaped


def flatten_weights(w):
    w_flat = np.array([])
    for iw in w:
        w_flat = np.concatenate([w_flat, iw.flatten()])
    return w_flat


def jacobian_list_to_matrix(dgdw_list):
    """ return shape: [ncells*nbasis_tensors, nweights] """
    dgdw = jacobian_cellwise_submatrices(dgdw_list)
    iw = dgdw.shape[-1]
    return dgdw.reshape([-1, iw])


def jacobian_cellwise_submatrices(dgdw_list):
    """ return shape: [ncells, nbasis_tensors, nweights] """
    ncells, nbasis_tensors = dgdw_list[0].shape[:2]
    nweights = 0
    for idgdw in dgdw_list:
        idgdw = idgdw.numpy()
        nweights += idgdw[0, 0].size
    dgdw = np.empty([ncells, nbasis_tensors, nweights])
    iw0 = 0
    for idgdw in dgdw_list:
        idgdw = idgdw.numpy()
        # flatten weights
        idgdw = idgdw.reshape([ncells, nbasis_tensors, -1])
        # number of weights
        iw = idgdw.shape[-1]
        # put into array
        dgdw[:, :, iw0:iw0+iw] = idgdw
        iw0 += iw
    assert nweights == iw0
    return dgdw
