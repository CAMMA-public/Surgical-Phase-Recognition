class Counter:
  def __init__(self, *args):
    self._counts = {}
    for key in args:
      self._counts[key] = 0


  def incr(self, *args):
    for key in args:
      self._counts[key] += 1


  def reset(self, *args):
    for key in args:
      self._counts[key] = 0


  def reset_all(self):
    for k in self._counts.keys():
      self._counts[k] = 0


  def __call__(self, key):
    return self._counts[key]


  def status(self):
    message = ""
    for k, v in self._counts.items():
      message += k + ": " + str(v) + " // "
    message = message[:-3]
    return message


  @property
  def counts(self):
    return self._counts
