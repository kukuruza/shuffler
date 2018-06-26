import os, sys, os.path as op
import numpy as np
import cv2
import logging
import simplejson as json
from utilities import bbox2roi, drawScoredRoi, drawScoredPolygon
from backendDb import deleteCar, carField
from backendImages import imread, maskread
from utilities import drawFrameId, drawMaskOnImage


def add_parsers(subparsers):
  displayParser(subparsers)


def displayParser(subparsers):
  parser = subparsers.add_parser('display',
    description='''Browse through database and see car bboxes on top of images.
                   Any key will scroll to the next image.''')
  parser.set_defaults(func=display)
  parser.add_argument('--winwidth', type=int, default=500)
  parser.add_argument('--labelmap_file',
      help='Json file with mappings from labels to color.'
           'Its content may be: "vis": [ {255: [0,128,128,30]} ]')
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--with_frameid', action='store_true', help='print frame number.')

def display (c, args):
  logging.info ('==== display ====')

  c.execute('SELECT imagefile,maskfile FROM images')
  image_entries = c.fetchall()
  logging.info('%d images found.' % len(image_entries))

  if args.shuffle:
    np.random.shuffle(image_entries)

  if args.labelmap_file:
    if not op.exists(args.labelmap_file):
      raise Exception('Labelmap file does not exist: %s' % args.labelmap_file)
    with open(args.labelmap_file) as f:
      labelmap = json.load(f)['vis']
  else:
    labelmap = {}

  for (imagefile, maskfile) in image_entries:
    logging.info ('Image %s' % imagefile)

    # Overlay the mask.
    frame = imread(imagefile)
    if maskfile is not None:
      mask = maskread(maskfile)
      frame = drawMaskOnImage(frame, mask, labelmap)
    else:
      logging.info('No mask file.')

    # Overlay frame id.
    if args.with_frameid:
      drawFrameId(frame, imagefile)

    scale = float(args.winwidth) / max(frame.shape[0:2])
    cv2.imshow('display', cv2.resize(frame[:,:,::-1], dsize=(0,0), fx=scale, fy=scale))
    if cv2.waitKey(-10) == 27:
      break
