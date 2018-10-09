"""Classes that define the generator, discriminator and the end-to-end
generator."""
from collections import OrderedDict
import tensorflow as tf
from bgan.component import Component
from bgan.utils.neuralnet import NeuralNet

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

        nets['main'] = NeuralNet(self.tensor_in, config['net_g']['main'],
                                 name='main')

        if config['net_g']['main'][-1][0] == 'reshape':
            preactivated = tf.reshape(nets['main'].layers[-2].preactivated,
                                      (-1, 28, 28, 1))
        else:
            preactivated = tf.reshape(nets['main'].layers[-1].preactivated,
                                      (-1, 28, 28, 1))

        return nets['main'].tensor_out, nets, preactivated

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

        nets['main'] = NeuralNet(self.tensor_in, config['net_g']['main'],
                                 name='main')

        return nets['main'].tensor_out, nets

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

        nets['main'] = NeuralNet(self.tensor_in, config['net_d']['main'],
                                 name='main')

        return nets['main'].tensor_out, nets
