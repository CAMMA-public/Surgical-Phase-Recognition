import tensorflow as tf


def multiclass(labels, predictions, metric):
  metric_per_class = [
    metric(l, p) for l, p in zip(labels, predictions)
  ]
  metric_vals = [x for x, _ in metric_per_class]
  metric_ops = [y for _, y in metric_per_class]
  return metric_vals, metric_ops


def make_confusion(labels, predictions, n_classes):
  batch_confusion = tf.confusion_matrix(
    labels,
    predictions,
    num_classes=n_classes,
    dtype=tf.int32
  )
  total_confusion = tf.get_variable(
    "confusion",
    shape=batch_confusion.shape,
    dtype=batch_confusion.dtype,
    initializer=tf.zeros_initializer(),
    collections=[tf.GraphKeys.LOCAL_VARIABLES]
  )
  confusion_op = tf.assign_add(total_confusion, batch_confusion)
  confusion_image = tf.expand_dims(total_confusion, axis=0)
  confusion_image = tf.expand_dims(confusion_image, axis=3)
  return confusion_image, confusion_op
