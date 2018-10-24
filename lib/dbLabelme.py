import os, sys, os.path as op
import numpy as np
import cv2
from lxml import etree as ET
import collections
import logging
from glob import glob
import shutil
import sqlite3
from progressbar import progressbar
from pprint import pformat
import re

from .backendImages import ImageryReader
from .util import drawScoredPolygon


def add_parsers(subparsers):
  importLabelmeParser(subparsers)
  importLabelmeObjectsParser(subparsers)



def _pointsOfPolygon (annotation):
  pts = annotation.find('polygon').findall('pt')
  xs = []
  ys = []
  for pt in pts:
    xs.append( int(pt.find('x').text) )
    ys.append( int(pt.find('y').text) )
  logging.debug('Parsed polygon xs=%s, ys=%s.' % (xs, ys))
  return xs, ys

def _isPolygonDegenerate(xs, ys):
  assert len(xs) == len(ys), (len(xs), len(ys))
  return len(xs) == 1 or len(xs) == 2 or min(xs) == max(xs) or min(ys) == max(ys)



def importLabelmeParser(subparsers):
  parser = subparsers.add_parser('importLabelme',
    description='Import LabelMe annotations for a db.')
  parser.set_defaults(func=importLabelme)
  parser.add_argument('--annotations_dir', required=True,
      help='Directory with xml files. '
      'Images should be already imported with "addPictures" or "addImages".')
  parser.add_argument('--with_display', action='store_true')

def importLabelme (c, args):
  if args.with_display:
    imreader = ImageryReader(args.rootdir)

  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()
  logging.info('Found %d images' % len(imagefiles))

  annotations_paths = os.listdir(args.annotations_dir)

  for imagefile, in progressbar(imagefiles):
    logging.info ('Processing imagefile: "%s"' % imagefile)

    # Find annotation files that match the imagefile.
    # There may be 1 or 2 dots because of some bug/feature in LabelMe.
    imagename = re.compile('0*%s[\.]{1,2}xml' % op.splitext(op.basename(jpgpath))[0])
    logging.debug('Will try to match %s' % regex)
    matches = [f for f in annotations_paths if re.match(regex, f)]
    if len(matches) == 0:
      logging.info('Annotation file does not exist: "%s". Skip image.' % annotation_file)
      continue
    elif len(matches) > 1:
      logging.warning('Found multiple files: %s' % pformat(matches))
    annotation_file = op.join(args.annotations_dir, matches[0])
    logging.info('Got a match %s' % annotation_file)

    # get dimensions
    c.execute('SELECT height,width FROM images WHERE imagefile=?', (imagefile,))
    sz = (height,width) = c.fetchone()

    if args.with_display:
      img = imreader.imread(imagefile)

    tree = ET.parse(annotation_file)
    for object_ in tree.getroot().findall('object'):

      # skip if it was deleted
      if object_.find('deleted').text == '1':
        continue

      # find the name of object.
      name = object_.find('name').text

      # get all the points
      xs, ys = _pointsOfPolygon(object_)

      # filter out degenerate polygons
      if _isPolygonDegenerate(xs, ys):
        logging.warning('Degenerate polygon %s,%s in %s' % (str(xs), str(ys), annotation_name))
        continue

      c.execute('INSERT INTO objects(imagefile,name) VALUES (?,?)', (imagefile, name))
      objectid = c.lastrowid
      for i in range(len(xs)):
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (?,?,?);',
          (objectid, xs[i], ys[i]))

      if args.with_display:
        pts = np.array([xs, ys], dtype=np.int32).transpose()
        drawScoredPolygon (img, pts, name, score=1)

    if args.with_display: 
      cv2.imshow('importLabelmeImages', img[:,:,::-1])
      if cv2.waitKey(-1) == 27:
        args.with_display = False
        cv2.destroyWindow('importLabelmeImages')



def importLabelmeObjectsParser(subparsers):
  parser = subparsers.add_parser('importLabelmeObjects',
    description='Import LabelMe annotations of objects. '
    'For each objectid in the db, will look for annotation in the form objectid.xml')
  parser.set_defaults(func=importLabelmeObjects)
  parser.add_argument('--annotations_dir', required=True,
    help='Directory with xml files.')
  parser.add_argument('--with_display', action='store_true')
  parser.add_argument('--keep_original_object_name', action='store_true',
    help='Do not update the object name from parsed xml.')
  parser.add_argument('--polygon_name',
    help='If specified, give each polygon entry this name.')

def importLabelmeObjects(c, args):
  if args.with_display:
    imreader = ImageryReader(rootdir=args.rootdir)

  annotations_paths = os.listdir(args.annotations_dir)

  c.execute('SELECT objectid,imagefile FROM objects')
  for objectid, imagefile in progressbar(c.fetchall()):
    logging.debug ('Processing object: %d' % objectid)

    # Find annotation files that match the object.
    # There may be 1 or 2 dots because of some bug/feature in LabelMe.
    regex = re.compile('0*%s[\.]{1,2}xml' % str(objectid))
    logging.debug('Will try to match %s' % regex)
    matches = [f for f in annotations_paths if re.match(regex, f)]
    if len(matches) == 0:
      logging.info('Annotation file does not exist: "%s". Skip image.' % annotation_file)
      continue
    elif len(matches) > 1:
      logging.warning('Found multiple files: %s' % pformat(matches))
    annotation_file = op.join(args.annotations_dir, matches[0])
    logging.info('Got a match %s' % annotation_file)

    tree = ET.parse(annotation_file)
    objects_ = tree.getroot().findall('object')

    # remove all deleted
    objects_ = [object_ for object_ in objects_ if object_.find('deleted').text != '1']
    if len(objects_) > 1:
      logging.error('More than one object in %s' % annotation_name)
      continue
    object_ = objects_[0]

    # find the name of object.
    name = object_.find('name').text
    if not args.keep_original_object_name:
      c.execute('UPDATE objects SET name=? WHERE objectid=?', (name, objectid))

    # get all the points
    xs, ys = _pointsOfPolygon(object_)

    # validate roi
    c.execute('SELECT height,width FROM images WHERE imagefile=?', (imagefile,))
    shape = (height,width) = c.fetchone()
    roi = [min(ys), min(xs), max(ys), max(xs)]
    if _isOutOfBorders(roi, shape):
      logging.warning ('roi %s out of borders: %s' % (str(roi), str(shape)))

    # Filter out degenerate polygons
    if _isPolygonDegenerate(xs, ys):
      logging.error('degenerate polygon %s,%s in %s' % (str(xs), str(ys), annotation_name))
      continue
          
    # Update polygon.
    for i in range(len(xs)):
      polygon = (objectid, xs[i], ys[i], args.polygon_name)
      c.execute('INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)', polygon)

    if args.with_display:
      img = imreader.imread(imagefile)
      pts = np.array([xs, ys], dtype=np.int32).transpose()
      drawScoredPolygon (img, pts, name, score=1)
      cv2.imshow('importLabelmeObjects', img[:,:,::-1])
      if cv2.waitKey(-1) == 27:
        args.with_display = False
        cv2.destroyWindow('importLabelmeObjects')
