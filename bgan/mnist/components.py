"""Classes for the components of the model, including the generator, the
discriminator and the refiner.
"""
from collections import OrderedDict
import tensorflow as tf
from bgan.utils.neuralnet import NeuralNet

class Component(object):
    """Base class for components."""
    def __init__(self, tensor_in, condition, slope_tensor=None):
        if not isinstance(tensor_in, tf.Tensor):
            raise TypeError("`tensor_in` must be of tf.Tensor type")

        self.tensor_in = tensor_in
        self.condition = condition
        self.slope_tensor = slope_tensor

        self.scope = None
        self.tensor_out = tensor_in
        self.nets = OrderedDict()
        self.vars = None

    def __repr__(self):
        return "Component({}, input_shape={}, output_shape={})".format(
            self.scope.name, self.tensor_in.get_shape(),
            str(self.tensor_out.get_shape()))

    def get_summary(self):
        """Return the summary string."""
        cleansed_nets = []
        for net in self.nets.values():
            if isinstance(net, NeuralNet):
                if net.scope is not None:
                    cleansed_nets.append(net)
            if isinstance(net, list):
                if net[0].scope is not None:
                    cleansed_nets.append(net[0])
        return '\n'.join(
            ["{:-^80}".format(' ' + self.scope.name + ' '),
             "{:39} {}".format('Input', self.tensor_in.get_shape())]
            + ['-' * 80 + '\n' + (x.get_summary()) for x in cleansed_nets])

class Generator(Component):
    """Class that defines the generator."""
    def __init__(self, tensor_in, config, condition=None, name='Generator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the generator."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_g']['shared'],
                                   name='shared')

        return nets['shared'].tensor_out, nets

class Discriminator(Component):
    """Class that defines the discriminator."""
    def __init__(self, tensor_in, config, condition=None, name='Discriminator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the discriminator."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_d']['shared'],
                                   name='shared')

        return nets['shared'].tensor_out, nets

class Refiner(Component):
    """Class that defines the refiner."""
    def __init__(self, tensor_in, config, condition=None, slope_tensor=None,
                 name='Refiner', reuse=None):
        super().__init__(tensor_in, condition, slope_tensor)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets, self.preactivated = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the refiner."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_r']['shared'],
                                   name='shared')

        return (nets['shared'].tensor_out, nets,
                nets['shared'].layers[-1].preactivated)

class End2EndGenerator(Component):
    """Class that defines the end-to-end generator."""
    def __init__(self, tensor_in, config, condition=None, slope_tensor=None,
                 name='End2EndGenerator', reuse=None):
        super().__init__(tensor_in, condition, slope_tensor)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets, self.preactivated = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the end-to-end generator."""
        nets = OrderedDict()

        nets['shared'] = NeuralNet(self.tensor_in, config['net_g']['shared'],
                                   name='shared')

        return (nets['shared'].tensor_out, nets,
                tf.reshape(nets['shared'].layers[-2].preactivated,
                           (-1, 28, 28, 1)))
