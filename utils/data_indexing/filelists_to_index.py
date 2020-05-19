import csv
import os
from glob import glob


def text_to_filelist(text):
  with open(text) as f:
      filelist = list(f.readlines())
      filelist = [x[:-1] for x in filelist]
  return filelist


filelist_dir = "./old_filelists"
dat_dir = "/mnt/xaware5_data1/home/yu/data/cholec120/frames"

os.chdir(filelist_dir)

vid_paths_list = [
    os.path.join(dat_dir, vid) for vid in os.listdir(dat_dir)
    if os.path.isdir(os.path.join(dat_dir, vid))
]
vid_names_list = [os.path.basename(x) for x in vid_paths_list]
vids_by_name = {k: v for v, k in enumerate(vid_names_list)}
n_vids = len(vid_names_list)

test_set_descriptors = [
  "cholec120_fold_0_testing.filelist",
  "cholec120_fold_1_testing.filelist",
  "cholec120_fold_2_testing.filelist",
  "cholec120_fold_3_testing.filelist"
]

eval_set_descriptors = [
  "cholec120_fold_0_validation.filelist",
  "cholec120_fold_1_validation.filelist",
  "cholec120_fold_2_validation.filelist",
  "cholec120_fold_3_validation.filelist"
]

train_set_descriptors = [
  "cholec120_fold_0_training.filelist",
  "cholec120_fold_1_training.filelist",
  "cholec120_fold_2_training.filelist",
  "cholec120_fold_3_training.filelist"
]

test_set_lists = [text_to_filelist(x) for x in test_set_descriptors]
eval_set_lists = [text_to_filelist(x) for x in eval_set_descriptors]
train_set_lists = [text_to_filelist(x) for x in train_set_descriptors]

im_lists = []
for vid_path in vid_paths_list:
  im_list = [
    os.path.join(vid_path, im_name) for im_name in sorted(os.listdir(vid_path))
    if im_name[-3:] == "png"
  ]
  im_lists.append(im_list)

vid_info_list = [
  {
    "vid_id": i,
    "length": len(im_lists[i]),
    "name": vid_name,
    "test_set": "",
    "eval_set": "",
    "train_set": ""
  }
  for i, vid_name in enumerate(vid_names_list)
]

ordered_keys = ["vid_id", "length", "name", "test_set", "eval_set", "train_set"]

for i, test_set in enumerate(test_set_lists):
  for vid_name in test_set:
    j = vids_by_name[vid_name]
    vid_info_list[j]["test_set"] += str(i)

for i, test_set in enumerate(eval_set_lists):
  for vid_name in test_set:
    j = vids_by_name[vid_name]
    vid_info_list[j]["eval_set"] += str(i)

for i, test_set in enumerate(train_set_lists):
  for vid_name in test_set:
    j = vids_by_name[vid_name]
    vid_info_list[j]["train_set"] += str(i)

os.chdir("..")

with open("cholec120_master_index.csv", "w") as f:
  file_writer = csv.writer(f)
  file_writer.writerow(ordered_keys)
  for vid_info in vid_info_list:
    line = [vid_info[k] for k in ordered_keys]
    file_writer.writerow(line)
