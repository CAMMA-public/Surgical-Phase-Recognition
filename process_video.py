'''
Author: Tong Yu
Copyright (c) University of Strasbourg. All Rights Reserved.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import global_helpers as gh
from models.model_17 import Model as ConvNet
from models.model_18 import Model as TmpNet


N_H = 256
N_W = 256
FPS = 25


def phase_recognition(vidpath):
  framestack = extract_frames(vidpath)
  features = cnn_forward_pass(framestack)
  predictions = temporal_forward_pass(features)
  return predictions


def phase_plot(phases):
  fig = plt.figure(figsize=(10, 2))
  ax = fig.add_subplot(111)
  ax.set_yticks([], [])
  ax.pcolormesh(phases, cmap="Set2")


def extract_frames(vidpath):
  res = extract_raw_frames(vidpath)
  res = preprocess(res)
  return res


def extract_raw_frames(vidpath):
  res = []
  count = 0
  vidcap = cv2.VideoCapture(vidpath)
  flag = True
  while flag:
    flag, frame = vidcap.read()
    if frame is not None:
      if count % FPS == 0:
        res.append(frame)
    else:
      break
    count += 1
  vidcap.release()
  return res


def preprocess(frames):
  h_in = frames[0].shape[0]
  w_in = frames[0].shape[1]
  center = w_in / 2
  radius = h_in / 2
  w_left = int(center - radius)
  w_right = int(center + radius)
  frames = [
    cv2.resize(f[:, w_left:w_right, ::-1], (N_H, N_W)) / 255.0
    for f in frames
  ]
  return frames


def cnn_forward_pass(frames):
  tf.reset_default_graph()
  features = []
  hp = gh.parse_hp("hparams/hp_225.yaml")
  m = ConvNet(hp)
  inp = tf.placeholder(dtype=tf.float32, shape=[None, N_H, N_W, 3])
  _ = m.forward_pass(inp)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m.load_pretrained_resnet(sess, "checkpoints/cnn/cnn.ckpt")
    fetches = {
      "features": m.fetches["cnn_output"]
    }
    for f in frames:
      ret = sess.run(
        fetches,
        feed_dict={
          m.fetches["train_flag"]: False,
          inp: np.expand_dims(f, axis=0)
        }
      )
      features.append(ret["features"])
  return features


def temporal_forward_pass(features):
  tf.reset_default_graph()
  hp = gh.parse_hp("hparams/hp_302.yaml")
  m = TmpNet(hp)
  n_t = len(features)
  features_in = np.concatenate(features)
  out = m.forward_pass(features_in)
  out = tf.expand_dims(out, 0)
  labels = tf.zeros(shape=[1, n_t], dtype=tf.int32)
  _, transition_matrix = tf.contrib.crf.crf_log_likelihood(
    out,
    labels,
    tf.constant([n_t])
  )
  predictions, viterbi_score = tf.contrib.crf.crf_decode(
    out,
    transition_matrix,
    tf.constant([n_t])
  )
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "checkpoints/temporal/temporal.ckpt")
    fetches = {
      "predictions": predictions
    }
    ret = sess.run(
      fetches,
      feed_dict={m.fetches["train_flag"]: False}
    )
  return ret["predictions"]
