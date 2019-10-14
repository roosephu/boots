import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Tensor


class TanhNormal(tfp.distributions.Normal):
    def _sample_n(self, n, seed=None):
        return tf.tanh(super()._sample_n(n, seed=seed))  #.clip_by_value(-0.999, 0.999)

    def _mean(self):
        return tf.tanh(super()._mean())

    def _log_prob(self, a: Tensor):
        u = tf.atanh(a)
        return super()._log_prob(u) - tf.log(tf.maximum(1 - a * a, 1e-6))

    def sample_with_log_prob(self, n=1, seed=None):
        u = tf.squeeze(super()._sample_n(n, seed=seed))
        a = tf.tanh(u)
        log_probs = tf.reduce_sum(super()._log_prob(u) - tf.log(tf.maximum(1 - a * a, 1e-6)), axis=1)
        return a, log_probs
