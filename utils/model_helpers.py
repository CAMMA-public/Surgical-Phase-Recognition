import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer


class ChronoInitializer(Initializer):
  def __init__(self, t_max=None, seed=None, dtype=tf.float32):
    self.t_max = t_max
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return tf.log(tf.random_uniform(shape, 1.0, self.t_max, dtype, seed=self.seed))

  def get_config(self):
    return {
      "t_max": self.t_max,
      "seed": self.seed,
      "dtype": self.dtype.name
    }
