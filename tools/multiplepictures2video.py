import os.path as op
import torch
import numpy as np
import cv2
from glob import glob
from argparse import ArgumentParser
import progressbar
import logging
import sys
sys.path.append("/home/evgeny/src/shuffler/lib")
from backendImages import imread, maskread, VideoWriter

def multiplepictures2video_parser():
  parser = ArgumentParser('Make a video, where each frame is a grid of 4 images from 4 sources.')
  parser.add_argument('-i', '--in_path_pattern', nargs='+', required=True,
      help='E.g. "mydir/*.dat" or "mydir/*.npy". Remember to mask * with quotes.')
  parser.add_argument('-o', '--out_videopath', required=True)
  parser.add_argument('--dryrun', action='store_true', help='Do not write anything.')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--imwidth', type=int, required=True)
  return parser

def multiplepictures2video(args):
  num_videos = len(args.in_path_pattern)
  assert num_videos in [2, 4], 'Only 2x1 or 2x2 grid for now.'
  
  if not args.dryrun:
    writer = VideoWriter(vimagefile=args.out_videopath, overwrite=args.overwrite, fps=args.fps)

  def processImage(imagefile):
    image = imread(imagefile)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
    scale = float(args.imwidth) / image.shape[1]
    image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
    return image

  imagefiles_by_video = [sorted(glob(x)) for x in args.in_path_pattern]
  for imagefiles in progressbar.progressbar(zip(*imagefiles_by_video)):
    images = [processImage(imagefile) for imagefile in imagefiles]
    if len(images) == 4:
      gridimage = np.vstack([np.hstack([images[0], images[1]]), np.hstack([images[2], images[3]])])
    elif len(images) == 2:
      gridimage = np.vstack([images[0], images[1]])
    if not args.dryrun:
      writer.imwrite(gridimage)


if __name__ == '__main__':
  parser = multiplepictures2video_parser()
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  multiplepictures2video(args)
