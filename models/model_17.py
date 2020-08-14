# *** id          : 17
# *** tag         : CNN
# *** precursors  : 4
# *** architecture: resnet_v2_50
# *** input_size  : [n_minibatch, 224, 224, 3]
# *** output_size : [n_outputs]

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
import os
import utils.global_helpers as gh


class Model:

  def __init__(self, hp):
    self._hp              = hp
    self._train_flag      = tf.placeholder_with_default(True, [], name="train_flag")
    self._fetches         = {}
    self._fc_w = tf.get_variable(
      "fc_w",
      shape=[self._hp.n_cnn_outputs, self._hp.n_classes],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer(0, self._hp.ini_fc)
    )
    self._fc_b = tf.get_variable(
      "fc_b",
      shape=[1, self._hp.n_classes],
      dtype=tf.float32,
      initializer=tf.constant_initializer(gh.safe_get(self._hp, "ini_bias", 0.0))
    )

    self._fetches["train_flag"]          = self._train_flag
    self._fetches["dense_layer_weights"] = self._fc_w
    self._fetches["dense_layer_bias"]    = self._fc_b

  # =========================================================================
  def forward_pass(self, frames, headless=False):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(
      weight_decay=self._hp.weight_decay,
      batch_norm_decay=self._hp.batch_norm_decay,
      batch_norm_epsilon=self._hp.batch_norm_epsilon,
      batch_norm_scale=bool(self._hp.batch_norm_scale),
    )):
      tether, _ = resnet_v2.resnet_v2_50(
        frames,
        num_classes=None,
        global_pool=True,
        is_training=self._train_flag
      )
    tether = tf.reshape(tether, [-1, self._hp.n_cnn_outputs])
    logits = tf.matmul(tether, self._fc_w) + self._fc_b
    self._fetches["cnn_output"] = tether
    if headless:
      return tether
    else:
      return logits

  # =========================================================================
  def load_pretrained_resnet(self, sess, ckpt_file):
    # init_fn = slim.assign_from_checkpoint_fn(
    #   gh.grab_ckpt(self._hp.dir_checkpoints),
    #   slim.get_model_variables("resnet_v2_50")
    # )
    init_fn = slim.assign_from_checkpoint_fn(
      ckpt_file,
      slim.get_model_variables("resnet_v2_50")
    )
    print("loaded " + ckpt_file)
    init_fn(sess)

  # =========================================================================
  def list_trainable_variables(self, listed_parts=None):
    model_parts = {
      "nothing": lambda: [],
      "fc"     : lambda: [self._fc_w, self._fc_b],
      "cnn"     : lambda: slim.get_trainable_variables("resnet_v2_50"),
      "conv1"   : lambda: slim.get_trainable_variables("resnet_v2_50/conv1"),
      "block1"  : lambda: slim.get_trainable_variables("resnet_v2_50/block1"),
      "block2"  : lambda: slim.get_trainable_variables("resnet_v2_50/block2"),
      "block3"  : lambda: slim.get_trainable_variables("resnet_v2_50/block3"),
      "block4"  : lambda: slim.get_trainable_variables("resnet_v2_50/block4"),
      "postnorm": lambda: slim.get_trainable_variables("resnet_v2_50/postnorm"),
      "all"    : lambda: tf.trainable_variables()
    }
    if listed_parts:
      lst = []
      for part in listed_parts:
        lst += model_parts[part]()
    else:
      lst = tf.trainable_variables()
    return lst

  # =========================================================================
  @property
  def fetches(self):
    return self._fetches
