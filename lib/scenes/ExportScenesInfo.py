#!/usr/bin/env python
import os, os.path as op
import logging
import argparse
import simplejson as json
from glob import glob
from pprint import pprint, pformat
from lib.scene import Pose, Video

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  scenes = {}

  # TODO: rewrite with lib.iterateScenes.py

  cam_dirs = glob('data/scenes/???')
  for cam_dir in cam_dirs:

    cam_id = op.basename(cam_dir)
    cam_info = json.load(open(op.join(cam_dir, 'camera.json')))
    cam_info['maps'] = []
    cam_info['poses'] = []

    for map_dir in glob(op.join(cam_dir, 'map*')):
      map_id = op.basename(map_dir)
      map_info = json.load(open(op.join(map_dir, 'map.json')))
      map_info['map_id'] = map_id
      cam_info['maps'].append(map_info)

    for pose_dir in glob(op.join(cam_dir, 'pose*')):
      pose_id = op.basename(pose_dir)
      pose_info = json.load(open(op.join(pose_dir, 'pose.json')))
      pose_info['pose_id'] = pose_id
      cam_info['poses'].append(pose_info)

    if op.exists(op.join(cam_dir, 'videos/videos.json')):
      videos_info = json.load(open(op.join(cam_dir, 'videos/videos.json')))
      for video_dir in glob(op.join(cam_dir, 'videos/*')):
        if op.isdir(video_dir):
          video_id = op.basename(video_dir)
          video_info = json.load(open(op.join(video_dir, 'video.json')))
          videos_info['videos'][video_id] = video_info
      cam_info['videos'] = videos_info['videos']

    scenes[cam_id] = cam_info
  
  with open('data/scenes/scenes.json', 'w') as f:
    f.write(json.dumps(scenes, sort_keys=True, indent=2))
