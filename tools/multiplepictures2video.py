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

def multiplemasks2video_parser():
  parser = ArgumentParser('Make a video, where each frame is a grid of 4 images from 4 sources.')
  parser.add_argument('-i', '--in_path_pattern', nargs='+', required=True,
      help='E.g. "mydir/*.dat" or "mydir/*.npy". Remember to mask * with quotes.')
  parser.add_argument('-o', '--out_videopath', required=True)
  parser.add_argument('--dryrun', action='store_true', help='Do not write anything.')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--imwidth', type=int, required=True)
  return parser

def multiplemasks2video(args):
  assert len(args.in_path_pattern) == 4, 'Only 2x2 grid for now.'

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
    gridimage = np.vstack([np.hstack([images[0], images[1]]), np.hstack([images[2], images[3]])])
    writer.imwrite(gridimage)


if __name__ == '__main__':
  parser = multiplemasks2video_parser()
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  multiplemasks2video(args)
