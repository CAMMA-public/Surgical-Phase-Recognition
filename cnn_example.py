import tensorflow as tf
import numpy as np
from models.model_17 import Model
from utils import global_helpers as gh


hp = gh.parse_hp("hparams/hp_225.yaml")
m = Model(hp)


inp = np.zeros([1, 256, 256, 3], np.float32)
out = m.forward_pass(inp)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  m.load_pretrained_resnet(sess, "checkpoints/cnn/cnn.ckpt")

  fetches = {
    "logits": out,
    "features": m.fetches["cnn_output"]
  }
  returns = sess.run(
    fetches,
    feed_dict={m.fetches["train_flag"]: True}
  )
  print("done")
