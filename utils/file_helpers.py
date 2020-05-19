from ruamel_yaml import YAML
from pprint import pprint


def filegrab(component_name):
  if component_name:
    return component_name.replace(".", "/") + ".py"
  else:
    return ""


def head_parser(filename):
  try:
    yaml = YAML(typ="safe", pure=True)
    with open(filename) as open_file:
      lines = [y for y in open_file.readlines() if y[0:6] == "# *** "]
      lines = [y[6:] for y in lines]
      strn = "".join(lines)
      metadata = yaml.load(strn)
    return metadata
  except FileNotFoundError:
    return {}


def shorten(string_in, n):
  return string_in[:n] + (len(string_in) > n) * "..."


def yaml_to_dict(yamlfile):
  yaml = YAML(typ="safe", pure=True)
  with open(yamlfile) as yaml_raw:
    dict_out = yaml.load(yaml_raw)
  return dict_out


def to_front(df, *args):
  for arg in args[::-1]:
    col = df[arg]
    df.drop(labels=arg, axis=1, inplace=True)
    df.insert(0, arg, col)
  return df


def prune_dict(dict_in, *args):
  return {k: dict_in.get(k) for k in args}


def shorten_run_log(dict_in):
  shortened = [
    {
      k: v
      if k in ["run_experiment", "run_meta"]
      else {u: v.get(u) for u in ["id", "tag"]}
      for k, v in d.items()
    }
    for d in dict_in
  ]
  return shortened
