import tensorflow as tf
import numpy as np
from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True, weight_initializer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight_initializer is None:
            init_range = np.sqrt(6.0 / (in_features + out_features))
            weight_initializer = tf.random_uniform_initializer(-init_range, init_range)

        self.use_bias = bias
        with self.scope:
            self.weight = Parameter(weight_initializer([in_features, out_features], dtype=tf.float32), name='weight')
            if bias:
                self.bias = Parameter(tf.zeros([out_features], dtype=tf.float32), name='bias')

    def build(self):
        self.op_input = tf.placeholder(dtype=tf.float32, shape=[None, self.in_features], name='input')
        self.op_output = self(self.op_input)

    def forward(self, x):
        shape = x.get_shape().as_list()
        if len(shape) > 2:
            y = tf.tensordot(x, self.weight, [[len(shape) - 1], [0]])
        else:
            y = tf.matmul(x, self.weight)
        if self.use_bias:
            y = y + self.bias
        return y

    def fast(self, x):
        x = x.dot(self.weight.numpy())
        if self.use_bias:
            x = x + self.bias.numpy()
        return x

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[i] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def fast(self, x):
        for module in self._modules.values():
            x = module.fast(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return tf.nn.relu(x)

    def fast(self, x: np.ndarray):
        return np.maximum(x, 0)


class Tanh(Module):
    def forward(self, x):
        return tf.nn.tanh(x)

    def fast(self, x: np.ndarray):
        return np.tanh(x)


class Dropout(Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        assert 0 <= drop_prob <= 1
        self.drop_prob = drop_prob
        self.training = False

    def forward(self, x):
        if self.drop_prob == 0:
            return x
        return tf.nn.dropout(x, 1 - self.drop_prob)

    def fast(self, x: np.ndarray):
        if self.drop_prob == 0:
            return x
        noise = np.random.rand(*x.shape) >= self.drop_prob
        return noise * x

    def extra_repr(self):
        return f'p={self.drop_prob}'


class Squeeze(Module):
    def __init__(self, axis=None):
        super().__init__()
        self._axis = axis

    def forward(self, x):
        return tf.squeeze(x, axis=self._axis)

    def fast(self, x):
        return x.squeeze(axis=self._axis)


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return tf.nn.leaky_relu(x, alpha=self.alpha)

    def extra_repr(self):
        return f"alpha={self.alpha:.6f}"


class LayerNorm(Module):
    """
        PyTorch has `eps` with default 1e-5, while TensorFlow uses 1e-12.
    """
    def __init__(self, normalized_shape, eps=1e-12, elementwise_affine=True):
        super().__init__()
        assert elementwise_affine, "partial implementation"
        self.normalized_shape = normalized_shape
        self.eps = eps

        with self.scope:
            self.scale = Parameter(tf.ones(normalized_shape), name='scale')
            self.offset = Parameter(tf.zeros(normalized_shape), name='offset')

    def forward(self, x):
        rank = x.shape.ndims
        axes = list(range(rank - len(self.normalized_shape), rank))
        mean, var = tf.nn.moments(x, axes=axes, keep_dims=True)

        return tf.nn.batch_normalization(x, mean, var, self.offset, self.scale, self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"
