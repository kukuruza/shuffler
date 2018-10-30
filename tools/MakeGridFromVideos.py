#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import torch
import numpy as np
import cv2
from glob import glob
from argparse import ArgumentParser
import progressbar
import logging
import sys

from lib.backendImages import VideoWriter, getVideoLength

def MakeGridFromVideos_parser():
  parser = ArgumentParser('Take a list of video files, and make a new one, '
    'where each frame is a grid of frames from the input videos.')
  parser.add_argument('-i', '--in_video_paths', nargs='+', required=True)
  parser.add_argument('-o', '--out_video_path', required=True)
  parser.add_argument('--dryrun', action='store_true', help='Do not write anything.')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--imwidth', type=int, required=True)
  return parser

def MakeGridFromVideos(args):
  num_videos = len(args.in_video_paths)
  if not num_videos in [2, 3, 4]:
    raise ValueError('Only 2x1, 3x1, or 2x2 grid for now.')
  
  if not args.dryrun:
    writer = VideoWriter(vimagefile=args.out_video_path, overwrite=args.overwrite, fps=args.fps)

  def processImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    scale = float(args.imwidth) / image.shape[1]
    image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
    image = image[:,:,::-1]  # VideoWriter changes RGB to GBR.
    return image

  handles = [cv2.VideoCapture(path) for path in args.in_video_paths]
  num_frames = 100000000
  for i, handle in enumerate(handles):
    if not handle:
      raise ValueError('Video failed to open: "%s"' % args.in_video_paths[i])
    num_frames = min(num_frames, getVideoLength(handle))

  for i in progressbar.progressbar(range(num_frames)):
    rets, images = zip(*[handle.read() for handle in handles])
    for ivideo,ret in enumerate(rets):
      if not ret:
        raise ValueError('Failed to read frame %d from video %s' %
          (i, args.in_video_paths[ivideo]))

    images = [processImage(image) for image in images]
    if len(images) == 4:
      gridimage = np.vstack([np.hstack([images[0], images[1]]), np.hstack([images[2], images[3]])])
    elif len(images) == 2:
      gridimage = np.hstack([images[0], images[1]])
    elif len(images) == 3:
      gridimage = np.hstack([images[0], images[1], images[2]])

    if not args.dryrun:
      writer.imwrite(gridimage)


if __name__ == '__main__':
  parser = MakeGridFromVideos_parser()
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()
  print (args)

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  MakeGridFromVideos(args)
