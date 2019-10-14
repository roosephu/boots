from typing import List
import numpy as np
import tensorflow as tf
import lunzi as lz
import lunzi.nn as nn


class GaussianNormalizer(nn.Module):
    def __init__(self, name: str, shape: List[int], eps=1e-8, verbose=False, decay=1.):  # batch_size x ...
        super().__init__()

        self.name = name
        self.shape = shape
        self.eps = eps
        self.decay = decay
        self._verbose = verbose

        with self.scope:
            self.op_mean = nn.Parameter(tf.zeros(shape, dtype=tf.float32), name='mean', trainable=False)
            self.op_std = nn.Parameter(tf.ones(shape, dtype=tf.float32), name='std', trainable=False)
            self.op_n = nn.Parameter(tf.zeros([], dtype=tf.float32), name='n', trainable=False)

    def extra_repr(self):
        return f'shape={self.shape}'

    def forward(self, x: lz.Tensor, inverse=False):
        if inverse:
            return x * self.op_std + self.op_mean
        return (x - self.op_mean) / tf.maximum(self.op_std, self.eps)

    def update(self, samples: np.ndarray):
        old_mean, old_std, old_n = self.op_mean.numpy(), self.op_std.numpy(), self.op_n.numpy()
        old_n *= self.decay
        samples = samples - old_mean

        m = samples.shape[0]
        delta = samples.mean(axis=0)
        new_n = old_n + m
        new_mean = old_mean + delta * m / new_n
        new_std = np.sqrt((old_std**2 * old_n + samples.var(axis=0) * m + delta**2 * old_n * m / new_n) / new_n)

        self.load_state_dict({'op_mean': new_mean, 'op_std': new_std, 'op_n': new_n})

    def fast(self, samples: np.ndarray, inverse=False) -> np.ndarray:
        mean, std = self.op_mean.numpy(), self.op_std.numpy()
        if inverse:
            return samples * std + mean
        return (samples - mean) / np.maximum(std, self.eps)


def get_expect_maximum(n):
    """
        https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
        https://math.stackexchange.com/questions/1884280/expectation-of-the-maximum-absolute-value-of-gaussian-random-variables
    """
    return np.sqrt(2 * np.log(n)) + 1. / np.sqrt(np.pi * np.log(n))


class Normalizer(GaussianNormalizer):
    """
        The normalization scheme is to assume that all numbers we see
        follow Gaussian distribution and we normalize `x` to
        `(x - mean) / mu`, such that `mean` is the statistical mean and
        `mu` is computed by assuming `normalized(x) < C`, where `C` is
        an estimation of expectation of maximum of absolute value of n
        iid Gaussian r.v..
    """
    pass


class Normalizers(nn.Module):
    """
        Normalization scheme:
            1. Policy: normalized states -> actions
            2. Model: (normalized states, actions) -> (normalized diffs)
    """

    def __init__(self, dim_action: int, dim_state: int, decay=1.):
        super().__init__()
        self.action = Normalizer('action', [dim_action], decay=decay)
        self.state = Normalizer('state', [dim_state], decay=decay)
        self.diff = Normalizer('diff', [dim_state], decay=decay)

    def update(self, dataset):
        self.state.update(dataset['state'])
        self.action.update(dataset['action'])
        self.diff.update(dataset['next_state'] - dataset['state'])

    def forward(self):
        pass


if __name__ == '__main__':
    with tf.Session() as sess:
        normalizer = Normalizer((1,))
        sess.run(tf.global_variables_initializer())

        data = []
        for updates in [[1.], [2.], [3., 4.], [5, 6, 7.]]:
            normalizer.update(np.array(updates))

            data.extend(updates)
            print(normalizer.op_mean, normalizer.op_std, np.mean(data), np.std(data))

