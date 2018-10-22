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

from .backendImages import ImageryReader
from .util import drawScoredPolygon


def add_parsers(subparsers):
  importLabelmeImagesParser(subparsers)
  importLabelmeObjectsParser(subparsers)


def _getAnnotationPath(name, annotations_dir):
  # Both files below may exist because of some bug in Labelme.
  for name_template in ['*%s.xml', '*%s..xml']:
    logging.debug('Trying to match name_template: %s' % name_template)
    pattern = op.join(annotations_dir, name_template % name)
    logging.debug('Will look for pattern: %s' % pattern)
    annotation_files = glob(pattern)
    if len(annotation_files) > 1:
      logging.warning('found multiple files: %s' % pformat(annotation_files))
    if len(annotation_files) >= 1:
      logging.debug('Found annotation_file: %s for %s' % (annotation_files[0], name))
      return annotation_files[0]

  return None


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

def _isOutOfBorders(roi, shape):
  ''' For some reason, width+1, height+1 happens. '''
  return roi[0] < 0 or roi[1] < 0 or roi[2] >= shape[0]+1 or roi[3] >= shape[1]+1

def importLabelmeImagesParser(subparsers):
  parser = subparsers.add_parser('importLabelmeImages',
    description='Import labelme annotations for a db.')
  parser.set_defaults(func=importLabelmeImages)
  parser.add_argument('--annotations_dir', required=True,
      help='Directory with xml files.')
  parser.add_argument('--with_display', action='store_true')

def importLabelmeImages (c, args):
  if args.with_display:
    imreader = ImageryReader(args.rootdir)

  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()
  logging.info('Found %d images' % len(imagefiles))

  for imagefile, in progressbar(imagefiles):
    logging.info ('Processing imagefile: "%s"' % imagefile)

    imagename = op.splitext(op.basename(jpgpath))[0]
    annotation_file = _getAnnotationPath(imagename, args.annotations_dir)
    if annotation_file is None:
      logging.error('Annotation file does not exist: "%s". Skip image.' % annotation_file)
      continue

    tree = ET.parse(annotation_file)

    # get dimensions
    c.execute('SELECT height,width FROM images WHERE imagefile=?', (imagefile,))
    sz = (height,width) = c.fetchone()

    if args.with_display:
      img = imreader.imread(imagefile)

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
        logging.info('Degenerate polygon %s,%s in %s' % (str(xs), str(ys), annotation_name))
        continue

      # Validate roi.
      roi = [min(ys), min(xs), max(ys), max(xs)]
      if _isOutOfBorders(roi, sz):
        logging.warning ('roi %s out of borders: %s' % (str(roi), str(sz)))

      c.execute('INSERT INTO objects(imagefile,name) VALUES (?,?)', (imagefile, name))
      objectid = c.lastrowid
      for i in range(len(xs)):
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (?,?,?);',
          (objectid, xs[i], ys[i]))

      if args.with_display:
        pts = np.array([xs, ys], dtype=np.int32).transpose()
        drawScoredPolygon (img, pts, name, score=1)

    if args.with_display: 
      cv2.imshow('importLabelmeImages', img)
      if cv2.waitKey(-1) == 27:
        args.with_display = False
        cv2.destroyWindow('importLabelmeImages')



def importLabelmeObjectsParser(subparsers):
  parser = subparsers.add_parser('importLabelmeObjects',
    description='Import labelme annotations of objects. '
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

  c.execute('SELECT objectid,imagefile FROM objects')
  for objectid, imagefile in progressbar(c.fetchall()):
    logging.debug ('Processing object: %d' % objectid)

    annotation_file = _getAnnotationPath(str(objectid), args.annotations_dir)
    if annotation_file is None:
      logging.error('Annotation file does not exist: "%s". Skip image.' % annotation_file)
      continue

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
      cv2.imshow('importLabelmeObjects', img)
      if cv2.waitKey(-1) == 27:
        args.with_display = False
        cv2.destroyWindow('importLabelmeObjects')
