#!/usr/bin/env python
import os, os.path as op
import logging
import argparse
from lib.labelMatches import labelMatches
from lib.scene import Pose


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--camera_id', required=True, type=str, default="cam572")
  parser.add_argument('--pose_id', type=int, default=0)
  parser.add_argument('--no_backup', action='store_true')
  parser.add_argument('--winsize1', type=int, default=500)
  parser.add_argument('--winsize2', type=int, default=500)
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  pose = Pose(camera_id=args.camera_id, pose_id=args.pose_id)

  frame = pose.load_example()
  satellite = pose.map.load_satellite()
  matches_path = op.join(pose.get_pose_dir(), 'matches-map%d.json' % pose.map_id)

  labelMatches (satellite, frame, matches_path,
      winsize1=args.winsize1, winsize2=args.winsize2,
      name1='map', name2='pose',
      backup_matches=not args.no_backup)
