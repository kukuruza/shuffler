#!/usr/bin/env python
import os, os.path as op
import logging
import argparse
import cv2
import numpy as np
from imageio import get_writer, imwrite
from lib.scene import Video
from lib.labelMatches import loadMatches, getGifFrame
from lib.warp import warpVideoToPose


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--camera_id', required=True, type=int)
  parser.add_argument('--video_id', required=True, type=str)
  parser.add_argument('--no_backup', action='store_true')
  parser.add_argument('--ransac', action='store_true')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  video = Video(camera_id=args.camera_id, video_id=args.video_id)

  # Load matches.
  video_name = op.splitext(args.video_id)[0]
  matches_path = op.join(video.get_video_dir(), 'matches-pose%d.json' % video.pose.pose_id)
  src_pts, dst_pts = loadMatches(matches_path, 'video', 'pose')

  # Compute video->pose homography.
  src_pts = src_pts.reshape(-1,1,2)
  dst_pts = dst_pts.reshape(-1,1,2)
  method = cv2.RANSAC if args.ransac else 0
  H_video_to_pose, _ = cv2.findHomography(src_pts, dst_pts, method=method)
  video['H_video_to_pose'] = H_video_to_pose.copy().reshape((-1,)).tolist()
  print 'H_video_to_pose:\n', H_video_to_pose

  # Save both video->pose and video->map homographies.
  video.save(backup=not args.no_backup)

  # Warp poseframe for nice visualization.
  videoframe = video.load_example()
  poseframe = video.pose.load_example()
  warped_poseframe = warpVideoToPose(poseframe, args.camera_id, args.video_id, reverse_direction=True)
  warped_path = op.join(video.get_video_dir(), 'poseframe-warped.gif')
  with get_writer(warped_path, mode='I') as writer:
    N = 15
    for i in range(N):
      writer.append_data(getGifFrame(warped_poseframe, videoframe, float(i) / N))
