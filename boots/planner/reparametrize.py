import tensorflow as tf
import tensorflow_probability as tfp
from boots import *
from functools import singledispatch


@singledispatch
def reparametrize(distribution, xi) -> tf.Tensor:
    raise NotImplementedError


@reparametrize.register(TanhNormal)
def _(distribution, xi):
    loc, scale = distribution.loc, distribution.scale
    return tf.tanh(loc + scale * xi)


@reparametrize.register(tfp.distributions.Normal)
def _(distribution, xi):
    loc, scale = distribution.loc, distribution.scale
    return loc + scale * xi
