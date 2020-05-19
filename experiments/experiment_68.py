# *** id     : 68
# *** tag    : CNN feature extraction
# *** model  : models.model_17
# *** feeder : feeders.feeder_14
# *** trainer: none
# *** probe  : none

import tensorflow as tf
import os
import subprocess as subp
from utils import global_helpers as gh
from feeders.feeder_14 import Feeder
from models.model_17 import Model


class Experiment:
  def __init__(self, dirname, hpfile, **kwargs):
    self._root         = dirname
    self._hp           = gh.parse_hp(hpfile)
    self._start_from   = kwargs.get("start_from")
    self._n_save_every = kwargs.get("n_save_every", 25)
    self._trigger      = [0]
    self._dirs         = {
      "tboard": os.path.join(self._root, "tensorboard_files"),
      "tmp"   : os.path.join(self._root, "tmp"),
      "target": os.path.join(os.environ["DATA_ROOT"], self._hp.target_dir)
    }
    self._filenames  = {
      "interact": os.path.join(self._dirs["tmp"], "interact.trigger"),
      "results" : os.path.join(self._root, "results.yaml")
    }
    self._ios = self.setup_directories()
    if kwargs.get("seed"):
      tf.set_random_seed(kwargs["seed"])


  def setup_directories(self):
    for dirname in self._dirs.values():
      subp.run(["mkdir", "-p", dirname])
    for filename in self._filenames.values():
      subp.run(["touch", filename])
    ios = {
      "interact": open(self._filenames["interact"], "r"),
      "results" : open(self._filenames["results"], "w")
    }
    return ios


  def run(self):
    # GRAPH BUILDING
    feeder = Feeder(self._hp)
    model = Model(self._hp)

    frames, n_total, frame_ids, phase_ids, end_of_vid = feeder.deliver_next()
    features = model.forward_pass(frames=frames, headless=True)

    with tf.Session() as sess:
      # INITIALIZATIONS
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      model.load_pretrained_resnet(sess)
      saver = tf.train.Saver()
      if self._start_from:
        saver.restore(sess, self._start_from)
      id_step = 0
      id_vid = 0
      # TRAIN LOOP
      location = os.path.join(self._dirs["target"], "%s-%03i" % ("tfr", id_vid))
      writer = tf.python_io.TFRecordWriter(location)
      while True:  # TODO: train_loop()
        try:
          train_fetches = {
            "extractions": features,
            "frame_ids"  : frame_ids,
            "n_total"    : n_total,
            "phase_ids"  : phase_ids,
            "end_of_vid" : end_of_vid
          }
          train_returns = sess.run(
            train_fetches,
            feed_dict={model.fetches["train_flag"]: False}
          )
          print("=============================================")
          print("step " + str(id_step) + ", vid " + str(id_vid))
          print("phase ids: " + str(train_returns["phase_ids"][:10]) + "...")
          for xtr, f_id, n_t, ph_id in zip(*list(train_returns.values())[:-1]):
            tfr_features = tf.train.Features(
              feature={
                "extraction": gh.mk_bytes_feat(xtr.tostring()),
                "n_total"   : gh.mk_int64_feat(n_t),
                "frame_id"  : gh.mk_int64_feat(f_id),
                "phase_id"  : gh.mk_int64_feat(ph_id)
              }
            )
            ex = tf.train.Example(features=tfr_features)
            writer.write(ex.SerializeToString())
          id_step += 1
          # END OF VIDEO
          if train_returns["end_of_vid"]:
            print("---------------> Video over")
            self.log_results(id_step, id_vid)
            id_vid += 1
            location = os.path.join(self._dirs["target"], "%s-%03i" % ("tfr", id_vid))
            writer = tf.python_io.TFRecordWriter(location)
        # END OF EPOCH
        except tf.errors.OutOfRangeError:
          print("---------------> All done")
          break
        # SAVE ROUTINE

    self.close_all_files()


  def log_results(self, step, video):  # TODO: add to global_helpers
    self._ios["results"].seek(0)
    self._ios["results"].write("vid: " + str(video) + "\n")
    self._ios["results"].write("step: " + str(step) + "\n")
    self._ios["results"].flush()
    os.fsync(self._ios["results"].fileno())


  def close_all_files(self):
    for v in self._ios.values():
      v.close()

  # def listen(self):  # TODO: ext event listener
  #   while True:
  #     time.sleep(30)

  # def console_output(self):  # TODO: console logger
