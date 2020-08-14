import tensorflow as tf
from tensorflow.python.client import device_lib
import datetime
from ruamel_yaml import YAML
import subprocess as subp
import logging
import os


def generate_summaries(prefix, tensor_list):
  summaries = [
    tf.summary.scalar(prefix + "_" + str(i), ts)
    for i, ts in enumerate(tensor_list)
  ]
  return summaries


def scalar_summaries(metrics_dict, prefix=""):
  summaries = []
  for k, v in metrics_dict.items():
    summaries.append(tf.summary.scalar(prefix + k, v))
  return summaries


def list_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def std_gpu_config():
  config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
  return config


def parse_hp(hpfile):
  yaml = YAML(typ="safe", pure=True)
  with open(hpfile) as hp_raw:
    hps = yaml.load(hp_raw)
  tf_hps = tf.contrib.training.HParams()
  for k, v in hps["default"].items():
    tf_hps.add_hparam(k, v)
  return tf_hps


def safe_get(hp, key, default_val=None):
  try:
    return hp.values()[key]
  except KeyError as e:
    msg = " --- DEFAULTING " + str(key) + " TO " + str(default_val)
    print(msg)
    logging.error(str(e) + "\n" + msg)
    return default_val


def table_summary(fieldnames, rows, tablename="Table"):
  body = tf.stack(rows)
  body = tf.as_string(body)
  fields = tf.constant([[x] for x in fieldnames])
  result = tf.concat([fields, body], axis=1)
  return tf.summary.text(tablename, result)


def generate_dirnames(root):
  dirnames = {
    "ckpt"  : os.path.join(root, "saved_checkpoints"),
    "pred"  : os.path.join(root, "predictions"),
    "tboard": os.path.join(root, "tensorboard_files"),
    "tmp"   : os.path.join(root, "tmp")
  }
  return dirnames


def generate_filenames(dirs, root):
  filenames = {
    "notes"    : os.path.join(root, "notes.txt"),
    "interact" : os.path.join(dirs["tmp"], "interact.trigger"),
    "results"  : os.path.join(root, "results.yaml"),
    "event_log": os.path.join(root, "event.log")
  }
  return filenames


def setup_directories(dirs, filenames):
  for dirname in dirs.values():
    subp.run(["mkdir", "-p", dirname])
  for filename in filenames.values():
    subp.run(["touch", filename])
  ios = {
    "notes"   : open(filenames["notes"], "r"),
    "interact": open(filenames["interact"], "r"),
    "results" : open(filenames["results"], "w")
  }
  logging.basicConfig(filename=filenames["event_log"], level=logging.INFO)
  return ios


def save(saver, sess, count, save_dir=".", every=1):
  if (count + 1) % every == 0:
    print("Saving...")
    ckpt_name = os.path.join(save_dir, "ep_" + str(count) + ".ckpt")
    saver.save(sess, ckpt_name)


def grab_ckpt(ckpt_dir, idx=0):
  files = [x.split(".", 1)[0] for x in os.listdir(ckpt_dir) if ".ckpt" in x]
  files = sorted(list(set(files)))
  return os.path.join(ckpt_dir, files[idx]) + ".ckpt"


def custom_summary(text, tag):
  text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
  meta = tf.SummaryMetadata()
  meta.plugin_data.plugin_name = "text"
  summary = tf.Summary()
  summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
  return summary


def progress_summary(counter, tag):
  input_text = str(datetime.datetime.now()) + " : " + counter.status()
  summary = custom_summary(input_text, tag)
  return summary


def log_results(file_handle, logs):
  logs["date"] = str(datetime.datetime.now())
  file_handle.seek(0)
  for k, v in logs.items():
    file_handle.write(k + ": " + str(v) + "\n")
  file_handle.flush()
  os.fsync(file_handle.fileno())


def close_all_files(file_handles):
  for v in file_handles.values():
    v.close()


def preview_results(dict_in, count=None, title=None, max_length=16, verbose=False):
  if verbose:
    print("=============================================")
    if title:
      print(title)
    if count:
      print(count.status())
    k_length = max([len(k) for k in dict_in.keys()])
    for k, v in dict_in.items():
      print(k.ljust(k_length) + ": " + str(v)[:max_length])


def device_setter(worker):
  setter = tf.train.replica_device_setter(
    worker_device=worker,
    ps_device="/cpu:0",
    ps_tasks=1
  )
  return setter


def mk_int64_feat(value):
  return tf.train.Feature(
    int64_list=tf.train.Int64List(value=[value])
  )


def mk_bytes_feat(value):
  return tf.train.Feature(
    bytes_list=tf.train.BytesList(value=[value])
  )