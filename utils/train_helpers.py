import tensorflow as tf
from pprint import pprint


_loss_dict = {
  "huber": tf.losses.huber_loss,
  "absolute_difference": tf.losses.absolute_difference,
  "mean_squared_error": tf.losses.mean_squared_error
}


_norm_dict = {
  "l2": tf.nn.l2_loss,
  "l1": tf.abs
}


_optimizer_dict = {
  "adam": tf.train.AdamOptimizer
}


def gradient_scaling(model, names=None, factors=None):
  if names and factors:
    assert(len(names) == len(factors))
    scaling_dictionary = {}
    for name, factor in zip(names, factors):
      scaling_dictionary.update({var: factor for var in model.list_trainable_variables(name)})
    return scaling_dictionary
  else:
    return {}


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    v = grad_and_vars[0][1]
    with tf.device(v.device):  # maybe not a good idea
      grad = tf.stack([g for g, _ in grad_and_vars], axis=0)
      grad = tf.reduce_mean(grad, axis=0)
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads


def average_buffered_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grad = tf.stack(list(grad_and_vars), axis=0)
    grad = tf.reduce_mean(grad, axis=0)
    average_grads.append(grad)
  return average_grads
