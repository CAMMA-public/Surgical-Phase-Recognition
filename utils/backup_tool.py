#!/usr/bin/env python

import os
import datetime
import subprocess as subp
import argparse


BKP_LOC = "yu@xjanus.local:/media/xjanus_data2/yu/backups"

dirs = {
  "xaware_runs" : "/mnt/xaware5_data1/home/yu/projects/coORdination/runs",
  "xaware_ckpts": "/mnt/xaware5_data1/home/yu/projects/coORdination/checkpoints",
  "xjanus_runs" : "yu@xjanus.local:/media/xjanus_data2/yu/projects/coORdination/runs",
  "xcore_runs"  : "yu@xcore.local:/media/expomri_data2/yu/projects/coORdination/runs",
  "hpc_runs"    : "tyu@hpc-login1.u-strasbg.fr:~/projects/coORdination/runs"
}

parser = argparse.ArgumentParser()
parser.add_argument("-b", help=str(dirs))
parser.add_argument("-all", help="backup all")
args = parser.parse_args()

dir_to_save = dirs.get(args.b)


def save_dir(dirname):
  bkp_name = "bkp-" + datetime.datetime.now().strftime("%F-%H-%M-%S-%f")
  tmp_dir = os.path.join("/tmp", bkp_name)
  archive = tmp_dir + ".tar.gz"
  subp.run(["mkdir", "-p", tmp_dir])
  subp.run([
    "rsync",
    "-aP",
    dirname,
    tmp_dir,
    "--exclude",
    "*saved_checkpoints",
    "--exclude"
    "*tensorboard_files"
  ])
  subp.run(["tar", "-czf", tmp_dir, archive])
  subp.run(["rsync", archive, BKP_LOC])


if dir_to_save:
  save_dir(args.b)
