import tensorflow as tf
import os
import warnings
import logging
import csv
import numpy as np
from glob import glob
from pprint import pformat
from utils.data_indexing import filelists
from collections import namedtuple


CHOLEC120_MASTER_INDEX = "utils/data_indexing/cholec120_master_index.csv"
CHOLEC40_RESPLIT = "utils/data_indexing/cholec40_resplit.csv"


def grab_full_filelist(data_dir, absolute=False):
  if absolute:
    data_path = data_dir
  else:
    data_path = os.path.join(os.environ["DATA_ROOT"], data_dir)
  globbed = glob(data_path + "/**", recursive=True)
  all_files = [x for x in globbed if os.path.isfile(x)]
  all_files.sort(key=lambda x: x[-10:])
  return all_files


def extract_ranges(filelist, ranges):
  extracted = [filelist[r[0]:r[1]] for r in ranges]
  return sum(extracted, [])


def grab_splits(data_dir, *args, **kwargs):
  full_list = grab_full_filelist(data_dir, absolute=kwargs.get("absolute", False))
  split_lists = [extract_ranges(full_list, arg) for arg in args]
  return tuple(split_lists)


def extract_from_master_index(data_dir, num, restriction=[], **kwargs):
  assert(restriction != [])
  index = kwargs.get("index", CHOLEC120_MASTER_INDEX)
  n_total = kwargs.get("n_total", 120)
  full_list = grab_full_filelist(data_dir, absolute=kwargs.get("absolute", False))
  with open(index) as f:
    vid_info_list = list(csv.DictReader(f))

  test_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["test_set"]]
  eval_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["eval_set"]]
  train_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["train_set"]]

  test_names = [d["name"] for d in vid_info_list if str(num) in d["test_set"]]
  eval_names = [d["name"] for d in vid_info_list if str(num) in d["eval_set"]]
  train_names = [d["name"] for d in vid_info_list if str(num) in d["train_set"]]

  test_names = [d["length"] for d in vid_info_list if str(num) in d["test_set"]]
  eval_names = [d["length"] for d in vid_info_list if str(num) in d["eval_set"]]
  train_names = [d["length"] for d in vid_info_list if str(num) in d["train_set"]]

  test_set = [full_list[j] for j in test_numbers]
  eval_set = [full_list[j] for j in eval_numbers]
  train_set = [full_list[j] for j in train_numbers]

  split_integrity_check(test_set, eval_set, train_set, n_total)
  split_integrity_check(test_names, eval_names, train_names, n_total)

  if restriction is not None:
    logging.info("!!!!!!!!!!LIMITING THE TRAIN SET")
    train_numbers = apply_restriction(train_numbers, restriction)
    train_set = [full_list[j] for j in train_numbers]

  logging.info("SPLIT NO " + str(num))
  logging.info("TEST_SET: " + "\n" + pformat(list(zip(test_numbers, test_names))))
  logging.info("EVAL_SET: " + "\n" + pformat(list(zip(eval_numbers, eval_names))))
  logging.info("TRAIN_SET: " + "\n" + pformat(list(zip(train_numbers, train_names))))

  return test_set, eval_set, train_set


def extract_and_mark_gt(data_dir, num, gt=[], **kwargs):
  index = kwargs.get("index", CHOLEC120_MASTER_INDEX)
  n_total = kwargs.get("n_total", 120)
  full_list = grab_full_filelist(data_dir, absolute=kwargs.get("absolute", False))
  with open(index) as f:
    vid_info_list = list(csv.DictReader(f))

  test_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["test_set"]]
  eval_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["eval_set"]]
  train_numbers = [int(d["vid_id"]) for d in vid_info_list if str(num) in d["train_set"]]

  test_names = [d["name"] for d in vid_info_list if str(num) in d["test_set"]]
  eval_names = [d["name"] for d in vid_info_list if str(num) in d["eval_set"]]
  train_names = [d["name"] for d in vid_info_list if str(num) in d["train_set"]]

  test_names = [d["length"] for d in vid_info_list if str(num) in d["test_set"]]
  eval_names = [d["length"] for d in vid_info_list if str(num) in d["eval_set"]]
  train_names = [d["length"] for d in vid_info_list if str(num) in d["train_set"]]

  test_set = [full_list[j] for j in test_numbers]
  eval_set = [full_list[j] for j in eval_numbers]
  train_set = [full_list[j] for j in train_numbers]

  split_integrity_check(test_set, eval_set, train_set, n_total)
  split_integrity_check(test_names, eval_names, train_names, n_total)

  choices = filelists.fl_index[gt]
  assert(set(choices).issubset(set(train_numbers)))
  marks = [1 if x in choices else 0 for x in train_numbers]
  gts = [(x, y) for x, y, z in zip(train_numbers, train_names, marks) if z]

  logging.info("SPLIT NO " + str(num))
  logging.info("TEST_SET: " + "\n" + pformat(list(zip(test_numbers, test_names))))
  logging.info("EVAL_SET: " + "\n" + pformat(list(zip(eval_numbers, eval_names))))
  logging.info("TRAIN_SET: " + "\n" + pformat(list(zip(train_numbers, train_names))))
  logging.info("GROUND_TRUTH: " + "\n" + pformat(gts))

  return test_set, eval_set, train_set, marks


def apply_restriction(train_numbers, restriction):
  if type(restriction) == int:
    return train_numbers[:restriction]
  elif type(restriction) == list:
    return train_numbers[restriction[0]:restriction[1]]
  elif type(restriction) == str:
    choices = filelists.fl_index[restriction]
    assert(set(choices).issubset(set(train_numbers)))
    res = [x for x in train_numbers if x in choices]
    return res


def split_integrity_check(a, b, c, n_total=120):
  assert(len(set(a) | set(b) | set(c)) == n_total)
  assert(len(set(a) & set(b) & set(c)) == 0)


def shrink_list(lst, shrink_spec):
  def aux_shrink_list(end, start=0):
    return lst[start:end]
  try:
    return aux_shrink_list(shrink_spec[1], shrink_spec[0])
  except TypeError:
    return aux_shrink_list(shrink_spec)


def examine_filelist(filelist=range(120)):
  index = CHOLEC120_MASTER_INDEX
  with open(index) as f:
    vid_info_list = list(csv.DictReader(f))
  Video = namedtuple("Video", "vid_id length name test_set eval_set train_set")
  videos = [Video(*v.values()) for v in vid_info_list if int(v["vid_id"]) in filelist]
  print(pformat(videos))


def minibatch(ds, n_minibatch, no_drop=False):
  if no_drop:
    return ds.batch(n_minibatch)
  else:
    return ds.apply(tf.contrib.data.batch_and_drop_remainder(n_minibatch))


def values_affine_transform(shift=0.5, scale=2.0):
  def transform_func_handle(frames, *args):
    frames = tf.subtract(frames, shift)
    frames = tf.multiply(frames, scale)
    return tuple([frames] + list(args))
  return transform_func_handle


def resampler(n_resize):
  def resample_func_handle(frames, *args):
    frames = tf.image.resize_bilinear(
      frames,
      [n_resize, n_resize],
      align_corners=False
    )
    return tuple([frames] + list(args))
  return resample_func_handle


def cropped_rotate(angle, n_offset, n_resize):
  def rotate_func_handle(frames, *args):
    frames = tf.contrib.image.rotate(
      frames,
      (angle * np.pi) / 180.0,
      interpolation="BILINEAR"
    )
    frames = tf.image.crop_to_bounding_box(
      frames, n_offset, n_offset, n_resize, n_resize
    )
    return tuple([frames] + list(args))
  return rotate_func_handle


def simple_crop(n_offset_x, n_offset_y, n_resize):
  def crop_func_handle(frames, *args):
    frames = tf.image.crop_to_bounding_box(
      frames, n_offset_x, n_offset_y, n_resize, n_resize
    )
    return tuple([frames] + list(args))
  return crop_func_handle


def flag_end(n_total, frame_id, no_drop=False, n_minibatch=None):
  if no_drop:
    return tf.equal(frame_id, n_total - 1)
  else:
    return tf.equal(frame_id, n_total - n_minibatch)


def flag_end_mxgpu(n_total, frame_id, n_gpu, n_minibatch=None):
  n_tail = 1 + tf.mod(n_total - 1, n_minibatch)
  n_head = n_total - n_tail - 1
  flag_1 = tf.greater(frame_id, n_head)
  flag_2 = tf.greater(frame_id, n_head - n_minibatch)
  return tf.cond(
    tf.greater_equal(n_tail, n_gpu),
    lambda: flag_1,
    lambda: flag_2
  )


def chain(*args):
  warnings.warn(
    "LEGACY CHAINING FUNCTION: USE DH.COMBO INSTEAD",
    DeprecationWarning
  )

  def chain_func_handle(*inner_args):
    res = inner_args
    for func in args:
      res = func(*inner_args)
    return res
  return chain_func_handle


def combo(*args):
  def combo_func_handle(*inner_args):
    res = inner_args
    for func in args:
      res = func(*res)
    return res
  return combo_func_handle


def reducer(reduce_max_list=[], reduce_any_list=[]):
  def reduction_func_handle(*args):
    arglist = list(args)
    for i in reduce_max_list:
      arglist[i] = tf.reduce_max(arglist[i])
    for j in reduce_any_list:
      arglist[j] = tf.reduce_any(arglist[j])
    return tuple(arglist)
  return reduction_func_handle


def discard_shorter_than(min_length):
  def discard_func_handle(frames, *args):
    return tf.shape(frames)[0] >= min_length
  return discard_func_handle


def discard_tail(min_length):
  def discard_func_handle(frames, n_total, frame_id, *args):
    limit = n_total - tf.mod(n_total, min_length)
    return tf.less(frame_id, limit)
  return discard_func_handle


def chain_concat(ds_list):
  big_ds = ds_list[0]
  for ds in ds_list[1:]:
    big_ds = big_ds.concatenate(ds)
  return big_ds


def crop_to_multiple(n):
  def crop_func_handle(*args):
    length = tf.shape(args[0])[0]
    n_crop = length - tf.mod(length, n)
    res = []
    for arg in args:
      res.append(arg[0:n_crop])
    return tuple(res)
  return crop_func_handle


# DEPRECATED

def resizer(n_resize):
  def resize_func_handle(frames, *args):
    frames = tf.image.resize_bilinear(
      frames,
      [n_resize, n_resize],
      align_corners=False
    )
    frames = tf.subtract(frames, 0.5)
    frames = tf.multiply(frames, 2.0)
    return tuple([frames] + list(args))
  return resize_func_handle
