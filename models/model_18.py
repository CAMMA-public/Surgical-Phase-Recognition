'''
Author: Tong Yu
CAMMA - University of Strasbourg
'''

import tensorflow as tf
import utils.global_helpers as gh


class Model:
  # =========================================================================
  def __init__(self, hp):
    self._hp              = hp
    self._train_flag      = tf.placeholder_with_default(1.0, [], name="train_flag")
    self._fetches         = {}
    self._cell_fw = self.make_cell(
      self._hp.dropout_lstm,
      self._hp.n_state,
      self._hp.n_lstm_layers
    )
    self._cell_bw = self.make_cell(
      self._hp.dropout_lstm,
      self._hp.n_state,
      self._hp.n_lstm_layers
    )
    self._fc_w = tf.get_variable(
      "fc_w",
      shape=[2 * self._hp.n_state, self._hp.n_classes],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer(0, self._hp.ini_fc)
    )
    self._fc_b = tf.get_variable(
      "fc_b",
      shape=[1, self._hp.n_classes],
      dtype=tf.float32,
      initializer=tf.zeros_initializer()
    )

    self._fetches["train_flag"]          = self._train_flag
    self._fetches["dense_layer_weights"] = self._fc_w
    self._fetches["dense_layer_bias"]    = self._fc_b

  # =========================================================================
  def forward_pass(self, features):
    features = tf.expand_dims(features, axis=0)
    features = tf.reshape(features, [1, -1, self._hp.n_features])
    bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
      self._cell_fw,
      self._cell_bw,
      features,
      dtype=tf.float32,
      swap_memory=True
    )
    output_series_fw, output_series_bw = bi_outputs
    output_series = tf.concat([output_series_fw, output_series_bw], axis=2)
    output_series = tf.reshape(output_series, [-1, 2 * self._hp.n_state])
    logits = tf.matmul(output_series, self._fc_w) + self._fc_b
    self._fetches["lstm_output"] = output_series
    return logits

  # =========================================================================
  def list_trainable_variables(self, listed_parts=None):
    model_parts = {
      "nothing": lambda: [],
      "fc"     : lambda: [self._fc_w, self._fc_b],
      "lstm"   : lambda: self._cell_fw.weights + self._cell_bw.weights,
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
  def make_cell(self, dropout, n_state, n_layers):
    lstm_keep_prob = 1.0 - self._train_flag * dropout
    use_peephole = gh.safe_get(self._hp, "use_peephole", 0)
    cell = tf.nn.rnn_cell.MultiRNNCell(
      [
        tf.nn.rnn_cell.DropoutWrapper(
          tf.contrib.rnn.LSTMBlockCell(
            n_state,
            use_peephole=bool(use_peephole)
          ),
          output_keep_prob=lstm_keep_prob
        )
        for _ in range(n_layers)
      ]
    )
    return cell

  # =========================================================================
  @property
  def fetches(self):
    return self._fetches
