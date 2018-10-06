import os, sys, os.path as op
import numpy as np
import cv2
import logging
import simplejson as json
from progressbar import progressbar
from .backendImages import imread, maskread, VideoWriter, PictureWriter
from .utilities import drawFrameId, drawMaskOnImage, loadLabelmap


def add_parsers(subparsers):
  writeVideoParser(subparsers)
  writePicturesParser(subparsers)


def writeVideoParser(subparsers):
  parser = subparsers.add_parser('writeVideo', description='Write video of "imagefile" entries.')
  parser.add_argument('--out_videopath', required=True)
  parser.add_argument('--fps', type=int, default=2)
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')
  parser.add_argument('--labels', nargs='+', type=int, default=[],
      help='highlight this labels in the mask.')
  parser.add_argument('--labelmap_path',
      help='Json file with mappings from labels to color.'
           'Its content may be: "vis": [ {255: [0,128,128,30]} ]')
  parser.add_argument('--alpha', type=float, default=0.5,
      help='Transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')
  parser.set_defaults(func=writeVideo)

def writeVideo (c, args):
  logging.info ('==== writeVideo ====')

  labelmap = loadLabelmap(args.labelmap_path) if args.labelmap_path else {}

  writer = VideoWriter(vimagefile=args.out_videopath, overwrite=args.overwrite, fps=args.fps)

  c.execute('SELECT imagefile,maskfile FROM images')
  for imagefile,maskfile in progressbar(c.fetchall()):
    frame = imread(imagefile).copy()

    # Draw the mask.
    if args.labelmap_path and maskfile is not None:
      mask = maskread(maskfile)
      frame = drawMaskOnImage(frame, mask, labelmap, args.alpha)

    # Display the frame id.
    if args.with_frameid:
      drawFrameId(frame, imagefile)

    writer.imwrite(frame)
  writer.close()


def writePicturesParser(subparsers):
  parser = subparsers.add_parser('writePictures', description='Write video of "imagefile" entries.')
  parser.add_argument('--out_dir', required=True)
  parser.add_argument('--labels', nargs='+', type=int, default=[],
      help='highlight this labels in the mask.')
  parser.add_argument('--labelmap_path',
      help='Json file with mappings from labels to color.'
           'Its content may be: "vis": [ {255: [0,128,128,30]} ]')
  parser.add_argument('--alpha', type=float, default=0.5,
      help='Transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')
  parser.set_defaults(func=writePictures)

def writePictures (c, args):
  logging.info ('==== writePictures ====')

  labelmap = loadLabelmap(args.labelmap_path) if args.labelmap_path else {}

  writer = PictureWriter()

  c.execute('SELECT imagefile,maskfile FROM images')
  for imagefile,maskfile in progressbar(c.fetchall()):
    frame = imread(imagefile)

    # Draw the mask.
    if args.labelmap_path and maskfile is not None:
      mask = maskread(maskfile)
      frame = drawMaskOnImage(frame, mask, labelmap, args.alpha)

    # Display the frame id.
    if args.with_frameid:
      drawFrameId(frame, imagefile)

    out_imagename = op.basename(imagefile)
    out_imagename = '%s.jpg' % out_imagename if not op.splitext(out_imagename)[1] else out_imagename
    out_imagefile = op.join(args.out_dir, out_imagename)
    writer.imwrite(frame, out_imagefile)
