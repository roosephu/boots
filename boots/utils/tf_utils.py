import tensorflow as tf
import numpy as np


def get_tf_config():
    gpu_frac = 1

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_frac,
        allow_growth=True,
    )
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True,
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )

    return config


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=tf.float32):
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer
