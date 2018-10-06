import os, sys, os.path as op
import numpy as np
import cv2
import logging
import simplejson as json
from .utilities import bbox2roi, drawScoredRoi, drawScoredPolygon
from .backendDb import deleteCar, carField
from .backendImages import imread, maskread
from .utilities import drawFrameId, drawMaskOnImage, loadLabelmap


def add_parsers(subparsers):
  displayParser(subparsers)


def displayParser(subparsers):
  parser = subparsers.add_parser('display',
    description='''Browse through database and see car bboxes on top of images.
                   Any key will scroll to the next image.''')
  parser.set_defaults(func=display)
  parser.add_argument('--winwidth', type=int, default=1000)
  parser.add_argument('--labelmap_path',
      help='Json file with mappings from labels to color.'
           'Its content may be: "vis": [ {255: [0,128,128,30]} ]')
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--alpha', type=float, default=0.5,
      help='Transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')

def display (c, args):
  logging.info ('==== display ====')

  c.execute('SELECT imagefile,maskfile FROM images')
  image_entries = c.fetchall()
  logging.info('%d images found.' % len(image_entries))

  if args.shuffle:
    np.random.shuffle(image_entries)

  labelmap = loadLabelmap(args.labelmap_path) if args.labelmap_path else {}

  for (imagefile, maskfile) in image_entries:
    logging.info ('Image %s' % imagefile)
    frame = imread(imagefile)

    # Overlay the mask.
    if args.labelmap_path and maskfile is not None:
      mask = maskread(maskfile)
      frame = drawMaskOnImage(frame, mask, labelmap, args.alpha)

    # Overlay frame id.
    if args.with_frameid:
      drawFrameId(frame, imagefile)

    scale = float(args.winwidth) / frame.shape[1]
    cv2.imshow('display', cv2.resize(frame[:,:,::-1], dsize=(0,0), fx=scale, fy=scale))
    if cv2.waitKey(-10) == 27:
      break
