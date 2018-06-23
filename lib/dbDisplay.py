import os, sys, os.path as op
import numpy as np
import cv2
import logging
from utilities import bbox2roi, drawScoredRoi, drawScoredPolygon
from backendDb import deleteCar, carField
from backendImages import imread, maskread
from utilities import relabelMask


def add_parsers(subparsers):
  displayParser(subparsers)


def displayParser(subparsers):
  parser = subparsers.add_parser('display',
    description='''Browse through database and see car bboxes on top of images.
                   Any key will scroll to the next image.''')
  parser.set_defaults(func=display)
  parser.add_argument('--winwidth', type=int, default=500)
  parser.add_argument('--masked', action='store_true',
      help='if mask exists, show only the foreground area.')
  parser.add_argument('--label', type=int, default=255)
  parser.add_argument('--shuffle', action='store_true')

def display (c, args):
  logging.info ('==== display ====')

  c.execute('SELECT imagefile,maskfile FROM images')
  image_entries = c.fetchall()
  logging.info('%d images found.' % len(image_entries))

  if args.shuffle:
    np.random.shuffle(image_entries)

  for (imagefile, maskfile) in image_entries:
    c.execute('SELECT * FROM cars WHERE imagefile=?', (imagefile,))
    car_entries = c.fetchall()
    logging.info ('%d cars found for %s' % (len(car_entries), imagefile))

    display = imread(imagefile)
    if args.masked and maskfile is not None:
      mask = maskread(maskfile)
      mask[mask != args.label] = 0
      mask[mask == args.label] = 255
      display[np.stack([mask, mask, mask], axis=2) == 0] /= 4
    display = display.copy()

    for car_entry in car_entries:
      carid = carField(car_entry, 'id')
      roi   = bbox2roi (carField(car_entry, 'bbox'))
      name =  carField(car_entry, 'name')
      score = carField(car_entry, 'score')

      if has_polygons:
        c.execute('SELECT x,y FROM polygons WHERE carid=?', (carid,))
        polygon = c.fetchall()
      if score is None: score = 1
      if has_polygons and polygon:
        logging.info ('polygon: %s, name: %s, score: %f' % (str(polygon), name, score))
        drawScoredPolygon(display, polygon, '', score)
      else:
        logging.info ('roi: %s, name: %s, score: %f' % (str(roi), name, score))
        drawScoredRoi (display, roi, '', score)

    scale = float(args.winwidth) / max(display.shape[0:2])
    cv2.imshow('display', cv2.resize(display[:,:,::-1], dsize=(0,0), fx=scale, fy=scale))
    if cv2.waitKey(-10) == 27:
      break
