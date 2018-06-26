import os, sys, os.path as op
import numpy as np
import cv2
import logging
import simplejson as json
from progressbar import progressbar
from backendImages import imread, maskread, VideoWriter
from utilities import drawFrameId, drawMaskOnImage


def add_parsers(subparsers):
  writeVideoParser(subparsers)


def writeVideoParser(subparsers):
  parser = subparsers.add_parser('writeVideo', description='Write video of "imagefile" entries.')
  parser.add_argument('--out_videofile', required=True)
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')
  parser.add_argument('--labels', nargs='+', type=int, default=[],
      help='highlight this labels in the mask.')
  parser.add_argument('--labelmap_file',
      help='Json file with mappings from labels to color.'
           'Its content may be: "vis": [ {255: [0,128,128,30]} ]')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')
  parser.set_defaults(func=writeVideo)

def writeVideo (c, args):
  logging.info ('==== writeVideo ====')

  if args.labelmap_file:
    if not op.exists(args.labelmap_file):
      raise Exception('Labelmap file does not exist: %s' % args.labelmap_file)
    with open(args.labelmap_file) as f:
      labelmap = json.load(f)['vis']
  else:
    labelmap = {}

  writer = VideoWriter(vimagefile=args.out_videofile, overwrite=args.overwrite, fps=args.fps)

  c.execute('SELECT imagefile,maskfile FROM images')
  for imagefile,maskfile in progressbar(c.fetchall()):

    # Draw the mask.
    frame = imread(imagefile)
    if maskfile is not None:
      mask = maskread(maskfile)
      frame = drawMaskOnImage(frame, mask, labelmap)
    else:
      logging.info('No mask file.')

    # Display the frame id.
    if args.with_frameid:
      drawFrameId(frame, imagefile)

    writer.imwrite(frame)
  writer.close()
