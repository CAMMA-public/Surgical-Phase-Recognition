import os


class HDRecorder:
  TMP_NAME = ".recording"

  def __init__(self, root_dir, tmp_dir):
    self._root = root_dir
    self._target_description = []
    self._tmp_path = os.path.join(tmp_dir, Recorder.TMP_NAME)
    self._recording = open(self._tmp_path, "w")


  def record(self, msgs, *args):
    self.update_target(list(args))
    for msg in msgs:
      self._recording.write("".join([str(y) + "," for y in msg])[:-1] + "\n")


  def update_target(self, description):
    if self._target_description != description:
      self._target_description = description


  def cut(self):
    self._recording.close()
    target_strings = [str(x) for x in self._target_description]
    target_path = os.path.join(self._root, *target_strings) + ".txt"
    dirname = os.path.dirname(target_path)
    if not os.path.isdir(dirname):
      os.makedirs(dirname)
    os.rename(self._tmp_path, target_path)
    self._recording = open(self._tmp_path, "w")


  def shutdown(self):
    self._recording.close()
    os.remove(self._tmp_path)


class Recorder:
  def __init__(self, root_dir, tmp_dir=None):
    self._root = root_dir
    self._target_description = []
    self._recording = []


  def record(self, msgs, *args):
    self.update_target(list(args))
    for msg in msgs:
      self._recording.append("".join([str(y) + "," for y in msg])[:-1] + "\n")


  def update_target(self, description):
    if self._target_description != description:
      self._target_description = description


  def cut(self):
    target_strings = [str(x) for x in self._target_description]
    target_path = os.path.join(self._root, *target_strings) + ".txt"
    dirname = os.path.dirname(target_path)
    if not os.path.isdir(dirname):
      os.makedirs(dirname)
    with open(target_path, "w+") as f:
      f.writelines(self._recording)
    self._recording.clear()


  def shutdown(self):
    self._recording.clear()
