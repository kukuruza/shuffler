import os, sys, os.path as op
import numpy as np
import sqlite3
import cv2
import logging
from glob import glob
from pprint import pformat
from progressbar import progressbar

from .backendDb import objectField, polygonField, deleteImage, deleteObject
from .backendImages import ImageryReader
from .utilities import drawScoredRoi, drawScoredPolygon, bbox2roi

def add_parsers(subparsers):
  filterImagesOfAnotherDbParser(subparsers)
  filterObjectsAtBorderParser(subparsers)
  filterObjectsByIntersectionParser(subparsers)
  filterObjectsByNameParser(subparsers)
  filterEmptyImagesParser(subparsers)
  filterObjectsByScoreParser(subparsers)
  filterObjectsSQLParser(subparsers)


def filterImagesOfAnotherDbParser(subparsers):
  parser = subparsers.add_parser('filterImagesOfAnotherDb',
      description='Remove images from the db that are not in the reference db.')
  parser.add_argument('--ref_db_file', required=True)
  parser.set_defaults(func=filterImagesOfAnotherDb)

def filterImagesOfAnotherDb (c, args):

  # Get all the imagefiles from the reference db.
  conn_ref = sqlite3.connect('file:%s?mode=ro' % args.ref_db_file, uri=True)
  c_ref = conn_ref.cursor()
  c_ref.execute('SELECT imagefile FROM images')
  imagefiles_ref = c_ref.fetchall()
  imagenames_ref = [op.basename(x) for x, in imagefiles_ref]
  logging.info ('Total %d images in ref.' % len(imagefiles_ref))
  conn_ref.close()

  # Get all the imagefiles from the main db.
  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()

  # Delete.
  imagefiles_del = [x for x, in imagefiles if op.basename(x) not in imagenames_ref]
  for imagefile_del in imagefiles_del:
    deleteImage(c, imagefile_del)

  # Get all the imagefiles from the main db.
  c.execute('SELECT COUNT(*) FROM images')
  count = c.fetchone()[0]
  logging.info('%d images left.' % count)



def filterObjectsAtBorderParser(subparsers):
  parser = subparsers.add_parser('filterObjectsAtBorder',
    description='Delete bboxes closer than "border_thresh" from border.')
  parser.set_defaults(func=filterObjectsAtBorder)
  parser.add_argument('--border_thresh', default=0.03,
    help='How far from border relative to image dimension an object can be to be filtered out.')
  parser.add_argument('--with_display', action='store_true',
    help='Until <Esc> key, images are displayed with objects on Border as red, and other as blue.')

def filterObjectsAtBorder (c, args):

  def isPolygonAtBorder (polygon_entries, width, height, border_thresh_perc):
    xs = [polygonField(p, 'x') for p in polygon_entries]
    ys = [polygonField(p, 'y') for p in polygon_entries]
    border_thresh = (height + width) / 2 * border_thresh_perc
    dist_to_border = min (xs, [width - x for x in xs], ys, [height - y for y in ys])
    num_too_close = sum([x < border_thresh for x in dist_to_border])
    return num_too_close >= 2

  def isRoiAtBorder (roi, width, height, border_thresh_perc):
    border_thresh = (height + width) / 2 * border_thresh_perc
    logging.debug('border_thresh: %f' % border_thresh)
    return min (roi[0], roi[1], height+1 - roi[2], width+1 - roi[3]) < border_thresh

  if args.with_display:
    imreader = ImageryReader(rootdir=args.rootdir)

  # For the reference.
  c.execute('SELECT COUNT(1) FROM objects')
  num_before = c.fetchone()[0]

  c.execute('SELECT imagefile FROM images')
  for imagefile, in progressbar(c.fetchall()):

    if args.with_display:
      image = imreader.imread(imagefile)

    c.execute('SELECT width,height FROM images WHERE imagefile=?', (imagefile,))
    (imwidth, imheight) = c.fetchone()

    c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
    object_entries = c.fetchall()
    logging.debug('%d objects found for %s' % (len(object_entries), imagefile))

    for object_entry in object_entries:
      for_deletion = False

      # Get all necessary entries.
      objectid = objectField(object_entry, 'objectid')
      roi = bbox2roi (objectField(object_entry, 'bbox'))
      c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid,))
      polygon_entries = c.fetchall()

      # Find if the polygon or the roi is at the border, polygon has preference over roi.
      if len(polygon_entries) > 0:
        if isPolygonAtBorder (polygon_entries, imwidth, imheight, args.border_thresh):
          logging.debug ('border polygon %s' % str(polygon))
          for_deletion = True
      elif roi is not None:
        if isRoiAtBorder(roi, imwidth, imheight, args.border_thresh):
          logging.debug ('border roi %s' % str(roi))
          for_deletion = True
      else:
        logging.error('Neither polygon, nor bbox is available for objectid %d' % objectid)

      # Draw polygon or roi.
      if args.with_display:
        if len(polygon_entries) > 0:
          polygon = [(polygonField(p, 'x'), polygonField(p, 'y')) for p in polygon_entries]
          drawScoredPolygon (image, polygon, score=(0 if for_deletion else 1))
        elif roi is not None:
          drawScoredRoi (image, roi, score=(0 if for_deletion else 1))

      # Delete if necessary
      if for_deletion:
        deleteObject (c, objectid)

    if args.with_display:
      cv2.imshow('filterObjectsAtBorder', image[:,:,::-1])
      key = cv2.waitKey(-1)
      if key == 27:
        args.with_display = False
        cv2.destroyWindow('filterObjectsAtBorder')

  # For the reference.
  c.execute('SELECT COUNT(1) FROM objects')
  num_after = c.fetchone()[0]
  logging.info('Deleted %d out of %d objects.' % (num_before-num_after, num_before))



def filterObjectsByIntersectionParser(subparsers):
  parser = subparsers.add_parser('filterObjectsByIntersection',
    description='Remove cars that have high intersection with other cars.')
  parser.set_defaults(func=filterObjectsByIntersection)
  parser.add_argument('--intersection_thresh', default=0.1, type=float,
    help='How much an object has to intersect with others to be filtered out.')
  #parser.add_argument('--expand_perc', type=float, default=0.1)
  #parser.add_argument('--target_ratio', type=float, default=0.75)  # h / w.
  #parser.add_argument('--keep_ratio', action='store_true')
  parser.add_argument('--with_display', action='store_true',
    help='Until <Esc> key, display how each object intersects others. '
    'Bad objects are shown as as red, good as blue, while the others as green.')

def filterObjectsByIntersection (c, args):
  '''If 'expand_bboxes' is set to True, a car is removed if the intersection
     of its expanded box and original boxes of other cars is too high.
  '''
  def getRoiIntesection(rioi1, roi2):
    dy = min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])
    dx = min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])
    if dy <= 0 or dx <= 0: return 0
    return dy * dx

  if args.with_display:
    imreader = ImageryReader(rootdir=args.rootdir)

  # For the reference.
  c.execute('SELECT COUNT(1) FROM objects')
  num_before = c.fetchone()[0]

  c.execute('SELECT imagefile FROM images')
  for imagefile, in progressbar(c.fetchall()):

    if args.with_display:
      image = imreader.imread(imagefile)

    c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
    object_entries = c.fetchall()
    logging.debug('%d objects found for %s' % (len(object_entries), imagefile))

    good_objects = np.ones(shape=len(object_entries), dtype=bool)
    for iobject1, object_entry1 in enumerate(object_entries):

      #roi1 = _expandCarBbox_(object_entry1, args)
      roi1 = objectField(object_entry1, 'roi')
      if roi1 is None:
        logging.error('No roi for objectid %d, intersection on polygons not implemented.')
        continue

      area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
      if area1 == 0:
        logging.warning('An object in %s has area 0. Will delete.' % imagefile)
        good_objects[iobject1] = False
        break

      for iobject2, object_entry2 in enumerate(object_entries):
        if iobject2 == iobject1:
          continue
        roi2 = objectField(object_entry2, 'roi')
        if roi2 is None:
          continue
        intersection = getRoiIntesection(roi1, roi2) / float(area1)
        if intersection > args.intersection_thresh:
          good_objects[iobject1] = False
          break

      if args.with_display:
        image = imreader.imread(imagefile)
        drawScoredRoi (image, roi1, score=(1 if good_objects[iobject1] else 0))
        for iobject2, object_entry2 in enumerate(object_entries):
          if iobject1 == iobject2:
            continue
          object2 = objectField(object_entry2, 'roi')
          if roi2 is None:
              continue
          drawScoredRoi (image, roi2, score=0.5)
        cv2.imshow('filterObjectsByIntersection', image[:,:,::-1])
        key = cv2.waitKey(-1)
        if key == 27:
          cv2.destroyWindow('filterObjectsByIntersection')
          args.with_display = 0

    for object_entry, is_object_good in zip(object_entries, good_objects):
      if not is_object_good:
        deleteObject(c, objectField(object_entry, 'objectid'))


def filterObjectsByNameParser(subparsers):
  parser = subparsers.add_parser('filterObjectsByName',
    description='Filter away car entries with unknown names.')
  parser.add_argument('--good_names', nargs='+',
    help='A list of object names to keep. Others will be filtered.')
  parser.set_defaults(func=filterObjectsByName)

def filterObjectsByName (c, args):
  good_names = ','.join(['"%s"' % x for x in args.good_names])
  logging.info('Will keep the folllowing object names: %s' % pformat(good_names))
  c.execute('SELECT objectid FROM objects WHERE name NOT IN (%s)' % good_names)
  for objectid, in progressbar(c.fetchall()):
    deleteObject(c, objectid)


def filterEmptyImagesParser(subparsers):
  parser = subparsers.add_parser('filterEmptyImages')
  parser.set_defaults(func=filterEmptyImages)

def filterEmptyImages(c, args):
  c.execute('SELECT imagefile FROM images WHERE imagefile NOT IN '
            '(SELECT imagefile FROM objects)')
  for imagefile, in progressbar(c.fetchall()):
    deleteImage(imagefile)


def filterObjectsByScoreParser(subparsers):
  parser = subparsers.add_parser('filterObjectsByScore',
    description='Delete all objects that have score less than "score_threshold".')
  parser.set_defaults(func=filterObjectsByScore)
  parser.add_argument('--score_threshold', type=float, default=0.5)

def filterObjectsByScore (c, args):
  c.execute('SELECT objectid FROM objects WHERE score < %f' % args.score_threshold)
  for objectid, in progressbar(c.fetchall()):
    deleteObject(c, objectid)


def filterObjectsSQLParser(subparsers):
  parser = subparsers.add_parser('filterObjectsSQL',
    description='Delete objects not matching image_constraint and car_constraint.')
  parser.set_defaults(func=filterObjectsSQL)
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--where',
    help='the SQL "where" clause for the JOIN of "images", "objects", and "properties" table.')
  group.add_argument('--sql',
    help='an arbitrary SQL clause that should query "objectid"')

def filterObjectsSQL (c, args):
  c.execute('SELECT COUNT(1) FROM objects')
  logging.info('Before filtering have %d objects.' % len(c.fetchall()))

  if args.where:
    c.execute('SELECT objects.objectid FROM objects '
              'INNER JOIN images ON images.imagefile=objects.imagefile '
              'INNER JOIN properties ON objects.objectid=properties.objectid '
              'WHERE (%s)' % args.where)
  if args.sql:
    c.execute(args.sql)

  objectids = c.fetchall()
  for objectid, in progressbar(objectids):
    deleteObject(c, objectid)
