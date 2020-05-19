import tensorflow as tf
import numpy as np
from models.model_18 import Model
from utils import global_helpers as gh


hp = gh.parse_hp("hparams/hp_302.yaml")
m = Model(hp)

inp = np.zeros([256, 2048], np.float32)  # time * channels
out = m.forward_pass(inp)
out = tf.expand_dims(out, 0)
labels = tf.zeros(shape=[1, 256], dtype=tf.int32)

_, transition_matrix = tf.contrib.crf.crf_log_likelihood(
  out,
  labels,
  tf.constant([256])
)

predictions, viterbi_score = tf.contrib.crf.crf_decode(
  out,
  transition_matrix,
  tf.constant([256])
)

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, "checkpoints/temporal/temporal.ckpt")
  fetches = {
    "predictions": predictions
  }
  returns = sess.run(
    fetches,
    feed_dict={m.fetches["train_flag"]: False}
  )
  print("done")
