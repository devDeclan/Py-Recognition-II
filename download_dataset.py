import os
from os import path
import sys
import argparse
import glob
import mmcv
import tqdm
import random
from lib.config import DATASET_ROOT, WORKERS, VIDEOS_ROOT, FRAMES_ROOT, ANNOTATIONS_ROOT
from multiprocessing import Pool, current_process

def download_dataset():
  # checking if dataset root exist otherwise create it
  if not path.exists(DATASET_ROOT):
    print("👾 creating folder {}".format(DATASET_ROOT))
    os.makedirs(DATASET_ROOT)

  # check if the dataset file has already been downloaded
  # otherwise download it
  # (ano get data like that😂)
  dataset_rar = path.join(DATASET_ROOT, "dataset.rar")
  if not path.exists(dataset_rar):
    print("👾 downloading dataset file")
    dataset_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", dataset_rar,
      dataset_url
    ])
    os.system(command)
    print("🤓 dataset file downloaded")
  else:
    print("👾 dataset file already exists, skipping download")

  # extract the videos
  print("👾 extracting dataset file")
  command = " ".join([
    "unrar", "x",
    dataset_rar, DATASET_ROOT,
    ">", "/dev/null",
    "&&", "mv",
    path.join(DATASET_ROOT, "UCF-101"),
    VIDEOS_ROOT
  ])
  os.system(command)
  print("🤓 dataset file downloaded and extracted")

  # check if the annotations file has already been download
  # otherwise download it
  annotations_zip = path.join(DATASET_ROOT, "annotations.zip")
  if not path.exists(annotations_zip):
    print("👾 downloading annotations file")
    annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", annotations_zip,
      annotations_url
    ])
    os.system(command)
    print("🤓 annotations file downloaded")
  else:
    print("👾 annotations file already exists, skipping download")

  # extract the annotations
  print("👾 extracting annotations file")
  command = " ".join([
    "unzip",
    "-q", annotations_zip,
    "-d", DATASET_ROOT,
    "&&", "mv",
    path.join(DATASET_ROOT, "ucfTrainTestlist"),
    ANNOTATIONS_ROOT
  ])
  os.system(command)
  print("🤓 annotations file downloaded and extracted")

def decode_video_to_frames(video_item):
  full_path, video_path, video_id = video_item
  video_name = video_path.split(".")[0]
  out_full_path = path.join(FRAMES_ROOT, video_name)
  try:
    os.mkdir(out_full_path)
  except OSError:
    pass
  video = mmcv.VideoReader(full_path)
  for i in tqdm(range(len(video))):
    if video[i] is not None:
      mmcv.imwrite(
        video[i],
        "{}/img_{:05d}.jpg".format(
          out_full_path,
          i + 1
        )
      )
    else:
      print(
        "😔 length inconsistent!"
        "early stop with {} out of {} frames".format(
          i + 1,
          len(video)
        )
      )
      break
  print("🤓 {} done with {} frames".format(video_name, len(video)))
  sys.stdout.flush()
  return True

def decode_videos_to_frames():
  # checking if frames root exist otherwise create it
  if not path.exists(FRAMES_ROOT):
    print("👾 creating folder {}".format(FRAMES_ROOT))
    os.makedirs(FRAMES_ROOT)

  # create sub directories for frame classes
  videos_path = path.join(DATASET_ROOT, "videos")
  classes = os.listdir(videos_path)
  for classname in classes:
    class_dir = path.join(FRAMES_ROOT, classname)
    if not path.isdir(class_dir):
      print("    👾 creating folder {}".format(class_dir))
      os.makedirs(class_dir)

  # read videos from folder
  fullpath_list = glob.glob(videos_path + "/*/*.avi")
  done_fullpath_list = glob.glob(FRAMES_ROOT + "/*/*")
  print("👾 total number of videos {}".format(len(fullpath_list)))
  print("👾 total number of decoded videos {}".format(len(done_fullpath_list)))
  fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
  print("👾 total number of undecoded videos {}".format(len(fullpath_list)))
  fullpath_list = list(fullpath_list)
  
  #
  video_list = list(
    map(
      lambda p: path.join("/".join(p.split("/")[-2:])),
      fullpath_list
    )
  )

  pool = Pool(WORKERS)
  pool.map(
    decode_video_to_frames,
    zip(
      fullpath_list,
      video_list,
      range(len(video_list))
    )
  )

def parse_splits():
  class_index = [
    x.strip().split()
    for x in open(
      path.join(ANNOTATIONS_ROOT, "classIndex.txt")
    )
  ]

  def line2rec(line):
    items = line.strip().split(" ")
    video = items[0].split(".")[0]
    video = "/".join(video.split("/")[-level:])
    label = class_mapping[items[0].split("/")[0]]
    return video, label

  splits = []
  for i in range(1, 4):
    train_list = [
      line2rec(x) for x in open(
        path.join(
          ANNOTATIONS_ROOT,
          "trainlist{:02d}.txt".format(i)
        )
      )
    ]
    test_list = [
      line2rec(x) for x in open(
        path.join(
          ANNOTATIONS_ROOT,
          "testlist{:02d}.txt".format(i)
        )
      )
    ]
    splits.append((train_list, test_list))
  return splits

def parse_directory():
  def key_func(x):
    return "/".join(x.split("/")[-2:])

  # Parse directories holding extracted frames from standard benchmarks
  print("👾 parse frames under folder {}".format(path))
  frame_folders = glob.glob(path.join(FRAMES_ROOT, "*", "*"))
  def count_files(directory, prefix_list):
    lst = os.listdir(directory)
    cnt_list = [len(fnmatch.filter(lst, x+"*")) for x in prefix_list]
    return cnt_list

  # check RGB
  frame_info = {}
  for i, f in enumerate(frame_folders):
    all_count = count_files(f, ("img_", "flow_x_", "flow_y_"))
    k = key_func(f)

    x_count = all_count[1]
    y_count = all_count[2]
    if x_count != y_count:
      raise ValueError(
        "😔 x and y direction of video {} have different number of flow images".format(f)
      )
    if i % 200 == 0:
      print("🤓 {} videos parsed".format(i))
      
  frame_info[k] = (f, all_count[0], x_count)
  print("🤓 frame folder analysis done")
  return frame_info

def build_split_list(split, frame_info, shuffle):

  def build_set_list(set_list):
    rgb_list, flow_list = list(), list()
    for item in set_list:
      if item[0] not in frame_info:
        # print("item:", item)
        continue
      elif frame_info[item[0]][1] > 0:
        rgb_cnt = frame_info[item[0]][1]
        flow_cnt = frame_info[item[0]][2]
        rgb_list.append('{} {} {}\n'.format(item[0], rgb_cnt, item[1]))
        flow_list.append('{} {} {}\n'.format(item[0], flow_cnt, item[1]))
      else:
        rgb_list.append('{} {}\n'.format(item[0], item[1]))
        flow_list.append('{} {}\n'.format(item[0], item[1]))
    if shuffle:
      random.shuffle(rgb_list)
      random.shuffle(flow_list)
    return rgb_list, flow_list

  train_rgb_list, train_flow_list = build_set_list(split[0])
  test_rgb_list, test_flow_list = build_set_list(split[1])
  return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)

def build_file_list():
  frame_info = parse_directory()
  split_tp = parse_splits()
  assert len(split_tp) == args.num_split

  for i, split in enumerate(split_tp):
    lists = build_split_list(
      split_tp[i],
      frame_info,
      shuffle = True
    )
    filename = "train_split_{}_frames.txt".format(i + 1)
    with open(
      path.join(ANNOTATIONS_ROOT, filename),
      "w"
    ) as f:
      f.writelines(lists[0][0])
    filename = 'valid_split_{}_frames.txt'.format(i + 1)
    with open(
      path.join(ANNOTATIONS_ROOT, filename),
      "w"
    ) as f:
      f.writelines(lists[0][1])

if __name__ == "__main__":
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "prepare UCF101 dataset")
  parser.add_argument("--download", action="store_true")
  parser.add_argument("--decode_video", action="store_true")
  parser.add_argument("--build_file_list", action="store_true")
  args = parser.parse_args()

  if args.download:
    print("🤖 downloading UCF101 dataset")
    download_dataset()
  if args.decode_video:
    print("🤖 decoding videos to frames")
    decode_videos_to_frames()
  if args.build_file_list:
    print("🤖 generating training files")
    build_file_list()
