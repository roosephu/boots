import tensorflow as tf

import lunzi as lz
from .module import Module


class _Loss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def _reduce(self, loss):
        if self.reduction == 'mean':
            loss = tf.reduce_mean(loss, axis=-1)
        elif self.reduction == 'sum':
            loss = tf.reduce_sum(loss, axis=-1)
        return loss


class L1Loss(_Loss):
    def forward(self, output: lz.Tensor, target: lz.Tensor):
        return self._reduce(tf.abs(output - target))


class MSELoss(_Loss):
    def forward(self, output: lz.Tensor, target: lz.Tensor):
        return self._reduce(tf.square(output - target))


class L2Loss(_Loss):
    """
        For $(a - b)^2$ please use `MSELoss`
    """
    def __init__(self, reduction='mean', average=True):
        super().__init__(reduction)
        self.average = average

    def forward(self, output: lz.Tensor, target: lz.Tensor):
        loss = tf.square(output - target)
        if self.average:
            loss = tf.reduce_mean(loss, axis=-1)
        else:
            loss = tf.reduce_sum(loss, axis=-1)
        return self._reduce(tf.sqrt(loss))

