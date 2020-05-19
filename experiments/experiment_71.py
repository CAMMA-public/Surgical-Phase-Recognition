# *** id        : 71
# *** tag       : test biLSTM + CRF phase ID
# *** precursors: 70
# *** model     : models.model_18
# *** feeder    : feeders.feeder_13
# *** trainer   : trainers.trainer_05
# *** probe     : probes.probe_01

import tensorflow as tf
import sys
from utils import global_helpers as gh
from utils.counter import Counter
from utils.recorder import Recorder
from feeders.feeder_13 import Feeder
from models.model_18 import Model
from probes.probe_01 import Probe


class Experiment:
  def __init__(self, dirname, hpfile, **kwargs):
    self._root          = dirname
    self._hp            = gh.parse_hp(hpfile)
    self._start_from    = kwargs.get("start_from")
    self._n_save_every  = kwargs.get("n_save_every", 25)
    self._dirs          = gh.generate_dirnames(self._root)
    self._files         = gh.generate_filenames(self._dirs, self._root)
    self._ios           = gh.setup_directories(self._dirs, self._files)
    self._train_count   = Counter("step", "minibatch", "vid", "ep")
    self._eval_count    = Counter("step", "minibatch", "vid")
    self._global_step   = tf.train.get_or_create_global_step()
    self._recorder      = Recorder(self._dirs["pred"], self._dirs["tmp"])

    if kwargs.get("seed"):
      tf.set_random_seed(kwargs["seed"])


  @property
  def train_count(self):
    return self._train_count


  @property
  def eval_count(self):
    return self._eval_count


  def run(self):
    feeder  = Feeder(self._hp)
    model   = Model(self._hp)
    probe   = Probe(self._hp)

    frames, n_total, frame_ids, phase_ids, end_of_vid = feeder.deliver_next()
    logits = model.forward_pass(frames)
    labels = tf.reshape(phase_ids, [-1])
    logits = tf.expand_dims(logits, 0)
    labels = tf.expand_dims(labels, 0)
    labels = tf.cast(labels, tf.int32)
    log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(
      logits,
      labels,
      tf.expand_dims(tf.shape(frames)[0], 0)
    )
    predictions, viterbi_score = tf.contrib.crf.crf_decode(
      logits,
      transition_matrix,
      tf.expand_dims(tf.shape(frames)[0], 0)
    )
    loss = - log_likelihood
    loss = tf.reshape(loss, [])

    metrics = probe.build_metrics(
      tf.reshape(labels, [-1]),
      tf.reshape(predictions, [-1])
    )

    confusion            = tf.cast(metrics.pop("confusion"), tf.float64)
    confusion_summary    = tf.summary.image("confusion_matrix", confusion)
    eval_summaries       = gh.scalar_summaries(metrics) + [confusion_summary]
    eval_summary_op      = tf.summary.merge(eval_summaries, name="eval_summarizer")

    summary_writer = tf.summary.FileWriter(self._dirs["tboard"], graph=tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      saver = tf.train.Saver()
      try:
        saver.restore(sess, self._start_from)
      except ValueError:
        print("Checkpoint missing")
        sys.exit(1)
      while True:
        try:
          eval_fetches = {
            "global_step"    : self._global_step,
            "eval_op"        : probe.fetches["update_op"],
            "n_total"        : n_total,
            "end_of_vid"     : end_of_vid,
            "predictions"    : tf.reshape(predictions, [-1]),
            "labels"         : tf.reshape(labels, [-1])
          }
          eval_returns = sess.run(
            eval_fetches,
            feed_dict={model.fetches["train_flag"]: False}
          )
          summary_writer.add_summary(gh.progress_summary(self.eval_count, "train_status"))
          self._recorder.record(
            zip(eval_returns["labels"], eval_returns["predictions"]),
            self.train_count("ep"),
            eval_returns["n_total"]
          )
          self.eval_count.incr("step", "minibatch")
          if eval_returns["end_of_vid"]:
            self._recorder.cut()
            self.eval_count.incr("vid")
            print("Video: " + str(self.eval_count("vid")))
        except tf.errors.OutOfRangeError:
          print("---------------> EVALUATION OVER")
          eval_end_fetches = {
            "eval_summary": eval_summary_op,
            "acc"         : metrics["global_accuracy"],
            "rec"         : metrics["global_recall"],
            "pre"         : metrics["global_precision"]
          }
          eval_end_returns = sess.run(eval_end_fetches)
          sess.run([probe.fetches["reset_op"]])
          self.eval_count.reset_all()
          summary_writer.add_summary(eval_end_returns.pop("eval_summary"), self.train_count("step"))
          eval_quick_logs = {**self._train_count.counts, **eval_end_returns}
          gh.log_results(self._ios["results"], eval_quick_logs)
          break
    self._recorder.shutdown()
    gh.close_all_files(self._ios)
