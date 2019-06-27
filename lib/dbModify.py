import os, sys, os.path as op
import numpy as np
import cv2
import logging
import sqlite3
import imageio
from math import ceil
from random import shuffle
from glob import glob
from pprint import pformat
from datetime import datetime
from progressbar import progressbar
from ast import literal_eval

from .backendDb import makeTimeString, deleteImage, objectField, createDb
from .backendMedia import getPictureSize, MediaReader
from .util import drawScoredRoi, roi2bbox, bboxes2polygons, polygons2bboxes
from .utilExpandBoxes import expandRoiToRatio, expandRoi


def add_parsers(subparsers):
  bboxesToPolygonsParser(subparsers)
  polygonsToBboxesParser(subparsers)
  sqlParser(subparsers)
  addVideoParser(subparsers)
  addPicturesParser(subparsers)
  headImagesParser(subparsers)
  tailImagesParser(subparsers)
  expandBoxesParser(subparsers)
  moveMediaParser(subparsers)
  moveRootdirParser(subparsers)
  addDbParser(subparsers)
  subtractDbParser(subparsers)
  splitDbParser(subparsers)
  mergeIntersectingObjectsParser(subparsers)
  renameObjectsParser(subparsers)
  resizeAnnotationsParser(subparsers)
  propertyToNameParser(subparsers)


def bboxesToPolygonsParser(subparsers):
  parser = subparsers.add_parser('bboxesToPolygons',
    description='If polygons dont exist for an object, '
      'create a rectangular polygon from the bounding box.')
  parser.set_defaults(func=bboxesToPolygons)

def bboxesToPolygons(c, args):
  c.execute('SELECT objectid FROM objects WHERE objectid NOT IN '
           '(SELECT objectid FROM polygons)')
  for objectid, in progressbar(c.fetchall()):
    bboxes2polygons(c, objectid)


def polygonsToBboxesParser(subparsers):
  parser = subparsers.add_parser('polygonsToBboxes',
    description='Update bounding box in the "objects" table with values from "polygons".')
  parser.set_defaults(func=polygonsToBboxes)

def polygonsToBboxes(c, args):
  c.execute('SELECT objectid FROM objects WHERE objectid NOT IN '
            '(SELECT DISTINCT(objectid) FROM polygons)')
  for objectid, in progressbar(c.fetchall()):
    polygons2bboxes(c, objectid)


def sqlParser(subparsers):
  parser = subparsers.add_parser('sql',
    description='Run SQL commands.'
    'Recorded paths will be made relative to "rootdir" argument.')
  parser.add_argument('sql', nargs='+',
    help='A list of SQL statements.')
  parser.set_defaults(func=sql)

def sql (c, args):
  for sql in args.sql:
    logging.info('Executing SQL command: %s' % sql)
    c.execute(sql)


def addVideoParser(subparsers):
  parser = subparsers.add_parser('addVideo',
      description='Import frames from a video into the database. '
      'Recorded paths will be made relative to "rootdir" argument.')
  parser.add_argument('--image_video_path', required=True)
  parser.add_argument('--mask_video_path')
  parser.set_defaults(func=addVideo)

def addVideo (c, args):

  # Check video paths.
  if not op.exists(args.image_video_path):
    raise FileNotFoundError('Image video does not exist at: %s' % args.image_video_path)
  if args.mask_video_path is not None and not op.exists(args.mask_video_path):
    raise FileNotFoundError('Mask video does not exist at: %s' % args.mask_video_path)

  # Check the length of image video with imageio
  image_video = imageio.get_reader(args.image_video_path)
  image_length = image_video.get_length()
  image = image_video.get_data(0)
  image_video.close()
  logging.info('Video has %d frames' % image_length)
  if image_length == 0:
    raise ValueError('The image video is empty.')

  # Check if masks agree with images.
  if args.mask_video_path is not None:
    mask_video = imageio.get_reader(args.image_video_path)
    mask_length = mask_video.get_length()
    mask = mask_video.get_data(0)
    mask_video.close()
    if image_length != mask_length:
      raise ValueError('Image length %d mask video length %d mismatch' % (image_length, mask_length))
    if image.shape[0:2] != mask.shape[0:2]:
      # When mismatched, a mask could be a CNN prediction of almost the same shape.
      logging.warning('Image size %s and mask size %s mismatch.' % (image.shape[1], mask.shape[1]))

  # Get the paths.
  image_video_rel_path = op.relpath(op.abspath(args.image_video_path), args.rootdir)
  if args.mask_video_path is not None:
    mask_video_rel_path = op.relpath(op.abspath(args.mask_video_path), args.rootdir)

  # Write to db.
  for iframe in progressbar(range(image_length)):
    height, width = image.shape[0:2]
    imagefile = op.join(image_video_rel_path, '%06d' % iframe)
    maskfile = op.join(mask_video_rel_path, '%06d' % iframe) if args.mask_video_path else None
    timestamp = makeTimeString(datetime.now())
    c.execute('INSERT INTO images('
      'imagefile, width, height, maskfile, timestamp) VALUES (?,?,?,?,?)',
      (imagefile, width, height, maskfile, timestamp))


def addPicturesParser(subparsers):
  parser = subparsers.add_parser('addPictures',
      description='Import picture files into the database.'
      'File names (without extentions) for images and masks must match.')
  parser.add_argument('--image_pattern', required=True,
      help='Wildcard pattern for image files. E.g. "my/path/images-\\*.jpg"'
      'Escape "*" with quotes or backslash.')
  parser.add_argument('--mask_pattern', default='/dummy',
      help='Wildcard pattern for image files. E.g. "my/path/masks-\\*.png"'
      'Escape "*" with quotes or backslash.')
  parser.set_defaults(func=addPictures)

def addPictures (c, args):

  # Collect a list of paths.
  image_paths = sorted(glob(args.image_pattern))
  logging.debug('image_paths:\n%s' % pformat(image_paths, indent=2))
  if not image_paths:
    raise IOError('Image files do not exist for the frame pattern: %s' % args.image_pattern)
  mask_paths = sorted(glob(args.mask_pattern))
  logging.debug('mask_paths:\n%s' % pformat(mask_paths, indent=2))

  def _nameWithoutExtension(x):
    ''' File name without extension is the shared part between images and masks. '''
    return op.splitext(op.basename(x))[0]

  def _matchPaths(A, B):
    ''' Groups paths in lists A and B into pairs, e.g. [(a1, b1), (a2, None), ...] '''
    A = {_nameWithoutExtension(path): path for path in A}
    B = {_nameWithoutExtension(path): path for path in B}
    return [(A[name], (B[name] if name in B else None)) for name in A]

  # Find correspondences between images and masks.
  pairs = _matchPaths(image_paths, mask_paths)
  logging.debug('Pairs:\n%s' % pformat(pairs, indent=2))
  logging.info('Found %d images, %d of them have masks.' %
      (len(pairs), len([x for x in pairs if x[1] is not None])))

  # Write to database.
  for image_path, mask_path in progressbar(pairs):
    height, width = getPictureSize(image_path)
    imagefile = op.relpath(op.abspath(image_path), args.rootdir)
    maskfile  = op.relpath(op.abspath(mask_path),  args.rootdir) if mask_path else None
    timestamp = makeTimeString(datetime.now())
    c.execute('INSERT INTO images('
      'imagefile, width, height, maskfile, timestamp) VALUES (?,?,?,?,?)',
      (imagefile, width, height, maskfile, timestamp))


def headImagesParser(subparsers):
  parser = subparsers.add_parser('headImages',
      description='Keep the first N image entries.')
  parser.add_argument('-n', required=True, type=int)
  parser.set_defaults(func=headImages)

def headImages (c, args):
  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()

  if len(imagefiles) < args.n:
    logging.info('Nothing to delete. Number of images is %d' % len(imagefiles))
    return

  for imagefile, in imagefiles[args.n:]:
    deleteImage(c, imagefile)


def tailImagesParser(subparsers):
  parser = subparsers.add_parser('tailImages',
      description='Keep the last N image entries.')
  parser.add_argument('-n', required=True, type=int)
  parser.set_defaults(func=tailImages)

def tailImages (c, args):
  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()

  if len(imagefiles) < args.n:
    logging.info('Nothing to delete. Number of images is %d' % len(imagefiles))
    return

  for imagefile, in imagefiles[:args.n]:
    deleteImage(c, imagefile)


def expandBoxesParser(subparsers):
  parser = subparsers.add_parser('expandBoxes',
    description='Expand bbox in the four directions. '
    'To expand polygons too, first convert them to roi, using "polygonsToRoi" tool.')
  parser.set_defaults(func=expandBoxes)
  parser.add_argument('--expand_perc', type=float, required=True)
  parser.add_argument('--target_ratio', type=float,
    help='If specified, expand to match this height/width ratio, '
    'and if that is less than "expand_perc", then expand more.')
  parser.add_argument('--with_display', action='store_true',
    help='Until <Esc> key, display old and new bounding box for each object.')

def expandBoxes (c, args):
  if args.with_display:
    imreader = MediaReader(rootdir=args.rootdir)

  c.execute('SELECT imagefile FROM images')
  for (imagefile,) in progressbar(c.fetchall()):

    c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
    object_entries = c.fetchall()
    logging.debug('Found %d objects for %s' % (len(object_entries), imagefile))

    if args.with_display:
      image = imreader.imread(imagefile)

    for object_entry in object_entries:
      objectid = objectField(object_entry, 'objectid')
      oldroi = objectField(object_entry, 'roi')
      if oldroi is None:  # Roi is not specified.
        continue
      if args.target_ratio:
        roi = expandRoiToRatio(oldroi, args.expand_perc, args.target_ratio)
      else:
        roi = expandRoi(oldroi, (args.expand_perc, args.expand_perc))
      c.execute('UPDATE objects SET x1=?, y1=?,width=?,height=? WHERE objectid=?',
                tuple(roi2bbox(roi) + [objectid]))

      if args.with_display:
        drawScoredRoi(image, oldroi, score=0)
        drawScoredRoi(image, roi, score=1)

    if args.with_display:
      cv2.imshow('expandBoxes', image[:,:,::-1])
      key = cv2.waitKey()
      if key == 27:
        cv2.destroyWindow('expandBoxes')
        args.with_display = False


def moveMediaParser(subparsers):
  parser = subparsers.add_parser('moveMedia',
    description='Change imagefile and maskfile.')
  parser.set_defaults(func=moveMedia)
  parser.add_argument('--where_image', default='TRUE',
    help='the SQL "where" clause for "images" table. '
    'E.g. to change imagefile of JPG pictures from directory "from/mydir" only, use: '
    '\'imagefile LIKE "from/mydir/%%"\'')
  parser.add_argument('--image_path',
    help='the directory for pictures OR video file of images')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file of masks')

def moveMedia (c, args):

  if args.image_path:
    logging.debug ('Moving image dir to: %s' % args.image_path)
    c.execute('SELECT imagefile FROM images WHERE (%s)' % args.where_image)
    imagefiles = c.fetchall()

    for oldfile, in progressbar(imagefiles):
      if oldfile is not None:
        newfile = op.join (args.image_path, op.basename(oldfile))
        c.execute('UPDATE images SET imagefile=? WHERE imagefile=?', (newfile, oldfile))
        c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?', (newfile, oldfile))

  if args.mask_path:
    logging.debug ('Moving mask dir to: %s' % args.mask_path)
    c.execute('SELECT maskfile FROM images WHERE (%s)' % args.where_image)
    maskfiles = c.fetchall()

    for oldfile, in progressbar(maskfiles):
      if oldfile is not None:
        newfile = op.join (args.mask_path, op.basename(oldfile))
        c.execute('UPDATE images SET maskfile=? WHERE maskfile=?', (newfile, oldfile))


def moveRootdirParser(subparsers):
  parser = subparsers.add_parser('moveRootdir',
    description='Change imagefile and maskfile entries to be relative to the provided rootdir.')
  parser.set_defaults(func=moveRootdir)
  parser.add_argument('newrootdir',
    help='All paths will be relative to the newrootdir.')

def moveRootdir (c, args):
  logging.info('Moving from rootdir %s to new rootdir %s' % (args.rootdir, args.newrootdir))
  relpath = op.relpath(args.rootdir, args.newrootdir)
  logging.info('The path of oldroot relative to newrootdir is %s' % relpath)

  c.execute('SELECT imagefile FROM images')
  for oldfile, in progressbar(c.fetchall()):
    if oldfile is not None:
      newfile = op.normpath(op.join(relpath, oldfile))
      c.execute('UPDATE images SET imagefile=? WHERE imagefile=?', (newfile, oldfile))
      c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?', (newfile, oldfile))

  c.execute('SELECT maskfile FROM images')
  for oldfile, in progressbar(c.fetchall()):
    if oldfile is not None:
      newfile = op.normpath(op.join(relpath, oldfile))
      c.execute('UPDATE images SET maskfile=? WHERE maskfile=?', (newfile, oldfile))


def addDbParser(subparsers):
  parser = subparsers.add_parser('addDb',
    description='Adds info from "db_file" to the current open database. Objects can be merged.'
    'Duplicate "imagefile" entries are ignore, but all associated objects are added.')
  parser.set_defaults(func=addDb)
  parser.add_argument('--db_file', required=True)
  parser.add_argument('--db_rootdir',
    help='If specified, imagefiles from add_db are considered relative to db_rootdir. '
    'They will be modified to be relative to rootdir of the active database.')
  parser.add_argument('--merge_duplicate_objects', action='store_true')

def addDb (c, args):
  c.execute('ATTACH ? AS "added"', (args.db_file,))
  c.execute('BEGIN TRANSACTION')

  # Copy images.
  c.execute('INSERT OR REPLACE INTO images SELECT * FROM added.images')

  # Find the max "match" in "matches" table. New matches will go after that value.
  c.execute('SELECT MAX(match) FROM matches')
  max_match = c.fetchone()[0]
  if max_match is None:
    max_match = 0
  
  # Copy all other tables.
  c.execute('SELECT * FROM added.objects')
  logging.info('Copying objects.')
  for object_entry_add in progressbar(c.fetchall()):
    objectid_add = objectField(object_entry_add, 'objectid')

    # Copy objects.
    c.execute('INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
      'VALUES (?,?,?,?,?,?,?)', object_entry_add[1:])
    objectid = c.lastrowid

    # Copy properties.
    c.execute('SELECT key,value FROM added.properties WHERE objectid=?', (objectid_add,))
    for key, value in c.fetchall():
      c.execute('INSERT INTO properties(objectid,key,value) VALUES (?,?,?)', (objectid,key,value))

    # Copy polygons.
    c.execute('SELECT x,y,name FROM added.polygons WHERE objectid=?', (objectid_add,))
    for x, y, name in c.fetchall():
      c.execute('INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)', (objectid,x,y,name))

    # Copy matches.
    c.execute('SELECT match FROM added.matches WHERE objectid=?', (objectid_add,))
    for match, in c.fetchall():
      c.execute('INSERT INTO matches(objectid,match) VALUES (?,?)', (objectid, match + max_match))

  # Change all the added imagefiles so that they are relative 
  # to args.rootdir instead of args.db_rootdir.
  if args.db_rootdir is not None:
    c.execute('SELECT imagefile,maskfile FROM added.images')
    for imagefile,maskfile in c.fetchall():
      imagefile_new = op.relpath(op.join(args.db_rootdir, imagefile), args.rootdir)
      if maskfile is None:
        maskfile_new = maskfile
      else:
        maskfile_new = op.relpath(op.join(args.db_rootdir, maskfile), args.rootdir)
      c.execute('UPDATE images SET imagefile=?, maskfile=? WHERE imagefile=?',
          (imagefile_new, maskfile_new, imagefile))
      c.execute('UPDATE objects SET imagefile=?, maskfile=? WHERE imagefile=?',
          (imagefile_new, maskfile_new, imagefile))

  c.execute('END TRANSACTION')
  c.execute('DETACH DATABASE "added"')


def subtractDbParser(subparsers):
  parser = subparsers.add_parser('subtractDb',
    description='Removes images present in "db_file" from the database. '
    'Objects in the removed images are removed too.')
  parser.set_defaults(func=subtractDb)
  parser.add_argument('--db_file', required=True)
  parser.add_argument('--db_rootdir',
    help='If specified, imagefiles from add_db are considered relative to db_rootdir. '
    'They will be modified to be relative to rootdir of the active database.')

def subtractDb (c, args):

  # Get all the subtracted imagefiles
  c.execute('ATTACH ? AS "subtracted"', (args.db_file,))
  c.execute('SELECT imagefile from subtracted.images')
  imagefiles_subtracted = c.fetchall()
  c.execute('DETACH DATABASE "subtracted"')

  for imagefile, in progressbar(imagefiles_subtracted):
    # Maybe, change rootdir of imagefile, so that it matches our dataset.
    if args.db_rootdir is not None:
      imagefile = op.relpath(op.join(args.db_rootdir, imagefile), args.rootdir)
    # If present, delete imagefile and all its objects.
    c.execute('SELECT COUNT(1) FROM images WHERE imagefile=?;', (imagefile,))
    if c.fetchone()[0] != 0:
      deleteImage(c, imagefile)



def splitDbParser(subparsers):
  parser = subparsers.add_parser('splitDb',
    description='Split a db into several sets (randomly or sequentially).')
  parser.set_defaults(func=splitDb)
  parser.add_argument('--out_dir', default='.',
    help='Common directory for all output databases')
  parser.add_argument('--out_names', required=True, nargs='+',
    help='Output database names with ".db" extension.')
  parser.add_argument('--out_fractions', required=True, nargs='+', type=float,
    help='Fractions to put to each output db. '
    'If sums to >1, last databases will be underfilled.')
  parser.add_argument('--randomly', action='store_true')
    
def splitDb (c, args):
  c.execute('SELECT imagefile FROM images ORDER BY imagefile')
  imagefiles = c.fetchall()
  if args.randomly:
    shuffle(imagefiles)

  assert len(args.out_names) == len(args.out_fractions), \
    'Sizes of not equal: %d != %d' % (len(args.out_names), len(args.out_fractions))

  current = 0
  for db_out_name, fraction in zip(args.out_names, args.out_fractions):
    logging.info((db_out_name, fraction))
    num_images_in_set = int(ceil(len(imagefiles) * fraction))
    next = min(current + num_images_in_set, len(imagefiles))

    # Create a database, and connect to it.
    # Not using ATTACH, because I don't want to commit to the open database.
    logging.info('Writing %d images to %s' % (num_images_in_set, db_out_name))
    db_out_path = op.join(args.out_dir, db_out_name)
    if op.exists(db_out_path):
      os.remove(db_out_path)
    conn_out = sqlite3.connect(db_out_path)
    createDb(conn_out)
    c_out = conn_out.cursor()

    for imagefile, in imagefiles[current : next]:

      c.execute('SELECT * FROM images WHERE imagefile=?', (imagefile,))
      c_out.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)', c.fetchone())

      c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
      for object_entry in c.fetchall():
        c_out.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?)', object_entry)

      c.execute('SELECT * FROM properties WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?)', (imagefile,))
      for property_entry in c.fetchall():
        c_out.execute('INSERT INTO properties VALUES (?,?,?,?)', property_entry)

      c.execute('SELECT * FROM polygons WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?)', (imagefile,))
      for polygon_entry in c.fetchall():
        c_out.execute('INSERT INTO polygons VALUES (?,?,?,?,?)', polygon_entry)

      c.execute('SELECT * FROM matches WHERE objectid IN '
        '(SELECT objectid FROM objects WHERE imagefile=?)', (imagefile,))
      for match_entry in c.fetchall():
        c_out.execute('INSERT INTO matches VALUES (?,?,?)', match_entry)

    current = next
    conn_out.commit()
    conn_out.close()



def _mergeNObjects(c, objectids):
  ' Merge N objects given by their objectids. '

  objectid_new = objectids[0]
  logging.debug('Merging objects %s into object %d' % (str(objectids), objectid_new))

  # No duplicates.
  if len(objectids) == 1:
    return

  # String from the list.
  objectids_str = (','.join([str(x) for x in objectids]))

  # Merge properties.
  for objectid in objectids[1:]:
    c.execute('UPDATE properties SET objectid=? WHERE objectid=?;', (objectid_new, objectid))
  c.execute('SELECT key,COUNT(DISTINCT(value)) FROM properties WHERE objectid=? GROUP BY key', (objectid_new,))
  for key, num in c.fetchall():
    if num > 1:
      logging.debug('Deleting %d duplicate properties for key=%s and objectid=%s' % (num, key, objectid_new))
      c.execute('DELETE FROM properties WHERE objectid=? AND key=?', (objectid_new, key))

  # Merge polygons by adding all the polygons together.
  c.execute('UPDATE polygons SET objectid=?, name=(IFNULL(name, "") || objectid) WHERE objectid IN (%s)' % 
      objectids_str, (objectid_new,))

  # Merge matches.
  # Delete matches between only matched objects.
  c.execute('DELETE FROM matches WHERE match NOT IN '
            '(SELECT DISTINCT(match) FROM matches WHERE objectid NOT IN (%s))' % objectids_str)
  # Merge matches from all merged to other objects.
  c.execute('UPDATE matches SET objectid=? WHERE objectid IN (%s);' % objectids_str, (objectid_new,))
  # Matches are not necessarily distinct.
  # TODO: Remove duplicate or particially duplicate matches.

  # Merge score.
  c.execute('SELECT DISTINCT(score) FROM objects WHERE objectid IN (%s) AND score IS NOT NULL' % objectids_str)
  scores = [x for x, in c.fetchall()]
  logging.debug('Merged objects %s have scores %s' % (objectids_str, scores))
  if len(scores) > 1:
    logging.debug('Several distinct score values (%s) for objectids (%s). Will set NULL.' % (str(names), objectids_str))
    c.execute('UPDATE objects SET score=NULL WHERE objectid=?', (objectid_new,))
  elif len(scores) == 1:
    logging.debug('Writing score=%s to objectid=%s.' % (scores[0], objectid_new))
    c.execute('UPDATE objects SET score=? WHERE objectid=?', (scores[0], objectid_new))
  else:
    logging.debug('No score found for objectid=%s.' % (objectid_new,))

  # Merge name.
  c.execute('SELECT DISTINCT(name) FROM objects WHERE objectid IN (%s) AND name IS NOT NULL' % objectids_str)
  names = [x for x, in c.fetchall()]
  logging.debug('Merged objects %s have names %s' % (objectids_str, names))
  if len(names) > 1:
    logging.debug('Several distinct name values (%s) for objectids (%s). Will set NULL.' % (str(names), objectids_str))
    c.execute('UPDATE objects SET name=NULL WHERE objectid=?;', (objectid_new,))
  elif len(names) == 1:
    logging.debug('Writing name=%s to objectid=%s.' % (names[0], objectid_new))
    c.execute('UPDATE objects SET name=? WHERE objectid=?;', (names[0], objectid_new))
  else:
    logging.debug('No name found for objectid=%s.' % (objectid_new,))

  # Delete all duplicates from "objects" table. (i.e. except for the 1st object).
  c.execute('DELETE FROM objects WHERE objectid IN (%s) AND objectid != ?' % objectids_str, (objectid_new,))
  return objectid_new


def mergeIntersectingObjectsParser(subparsers):
  parser = subparsers.add_parser('mergeIntersectingObjects',
    description='Merge objects that intersect. '
    'Currently only pairwise, does not merge groups. That is left for future. '
    'Currently implements only intersection by bounding boxes.')
  parser.set_defaults(func=mergeIntersectingObjects)
  parser.add_argument('--IoU_threshold', type=float, default=0.5,
    help='Intersection over Union threshold to consider merging.')
  parser.add_argument('--target_name',
    help='Name to assign to merged objects. If not specified, '
    'the name is assigned only if the names of the two merged objects match.')
  parser.add_argument('--where_object1', default='TRUE',
    help='SQL "where" clause for the "from" objects. The default is any object. '
    'Objects "from" are merged with objects "to".')
  parser.add_argument('--where_object2', default='TRUE',
    help='SQL "where" clause for the "to" objects. The default is any object. '
    'Objects "from" are merged with objects "to".')
  parser.add_argument('--where_image', default='TRUE',
    help='the SQL "where" clause for "images" table. '
    'E.g. to change imagefile of JPG pictures from directory "from/mydir" only, use: '
    '\'imagefile LIKE "from/mydir/%%"\'')
  parser.add_argument('--with_display', action='store_true',
    help='Until <Esc> key, display merged opjects.')
    
def mergeIntersectingObjects (c, args):

  def _getIoU(roi1, roi2):
    ' Computes intersection over union for two rectangles. '
    intersection_y = max(0, (min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])))
    intersection_x = max(0, (min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])))
    intersection = intersection_x * intersection_y
    area1 = (roi1[3] - roi1[1]) * (roi1[2] - roi1[0])
    area2 = (roi2[3] - roi2[1]) * (roi2[2] - roi2[0])
    union = area1 + area2 - intersection
    IoU = intersection / union if union > 0 else 0.
    return IoU

  if args.with_display:
    imreader = MediaReader(rootdir=args.rootdir)

  # Make polygons if necessary
  logging.info('Syncing bboxes and polygons.')
  polygonsToBboxes(c, None)
  bboxesToPolygons(c, None)

  c.execute('SELECT imagefile FROM images WHERE (%s)' % args.where_image)
  for imagefile, in progressbar(c.fetchall()):
    logging.debug('Processing imagefile %s' % imagefile)

    # Get objects of interest from imagefile.
    c.execute('SELECT * FROM objects WHERE imagefile=? AND (%s)' 
        % args.where_object1, (imagefile,))
    objects1 = c.fetchall()
    c.execute('SELECT * FROM objects WHERE imagefile=? AND (%s)' 
        % args.where_object2, (imagefile,))
    objects2 = c.fetchall()
    logging.debug('Image %s has %d and %d objects to match' % 
      (imagefile, len(objects1), len(objects2)))

    # Compute pairwise distances between rectangles.
    pairwise_IoU = np.zeros(shape=(len(objects1), len(objects2)), dtype=float)
    for i1, object1 in enumerate(objects1):
      for i2, object2 in enumerate(objects2):
        # Do not merge an object with itself.
        if objectField(object1, 'objectid') == objectField(object2, 'objectid'):
          pairwise_IoU[i1, i2] = np.inf
        else:
          roi1 = objectField(object1, 'roi')
          roi2 = objectField(object2, 'roi')
          pairwise_IoU[i1, i2] = _getIoU(roi1, roi2)
    logging.debug('Pairwise_IoU is:\n%s' % pformat(pairwise_IoU))

    # Greedy search for pairs.
    pairs_to_merge = []
    for step in range(min(len(objects1), len(objects2))):
      i1, i2 = np.unravel_index(np.argmax(pairwise_IoU), pairwise_IoU.shape)
      logging.debug('Maximum reached at indices [%d, %d]' % (i1, i2))
      IoU = pairwise_IoU[i1, i2]
      # Stop if no more good pairs.
      if IoU < args.IoU_threshold:
        break
      # Disable these objects for the next step.
      pairwise_IoU[i1, :] = 0.
      pairwise_IoU[:, i2] = 0.
      # Add a pair to the list.
      pairs_to_merge.append([i1, i2])
      logging.info('Will merge objects %d (%s) and %d (%s) with IoU %f.' %
        (objectField(objects1[i1], 'objectid'), objectField(objects1[i1], 'name'),
         objectField(objects2[i2], 'objectid'), objectField(objects2[i2], 'name'), IoU))

    if args.with_display and len(pairs_to_merge) > 0:
      image = imreader.imread(imagefile)

    # Merge pairs.
    for i1, i2 in pairs_to_merge:
      objectid1 = objectField(objects1[i1], 'objectid')
      objectid2 = objectField(objects2[i2], 'objectid')

      old_roi1 = objectField(objects1[i1], 'roi')
      old_roi2 = objectField(objects2[i2], 'roi')

      if args.with_display:
        drawScoredRoi(image, objectField(objects1[i1], 'roi'), score=0)
        drawScoredRoi(image, objectField(objects2[i2], 'roi'), score=0.25)

      # Change the name to the target first.
      if args.target_name is not None:
        c.execute('UPDATE objects SET name=? WHERE objectid IN (?,?);',
          (args.target_name, objectid1, objectid2))

      #logging.info('Merging objects %d and %d.' % (objectid1, objectid2))
      new_objectid = _mergeNObjects(c, [objectid1, objectid2])
      polygons2bboxes(c, new_objectid)

      c.execute('SELECT * FROM objects WHERE objectid=?', (new_objectid,))
      new_roi = objectField(c.fetchone(), 'roi')
      logging.info('Merged ROIs %s and %s to make one %s' % (old_roi1, old_roi2, new_roi))

      if args.with_display:
        drawScoredRoi(image, new_roi, score=1)

    if args.with_display and len(pairs_to_merge) > 0:
      logging.getLogger().handlers[0].flush()
      cv2.imshow('mergeIntersectingObjects', image[:,:,::-1])
      key = cv2.waitKey()
      if key == 27:
        cv2.destroyWindow('mergeIntersectingObjects')
        args.with_display = False


def renameObjectsParser(subparsers):
  parser = subparsers.add_parser('renameObjects',
    description='Map object names. Can delete names not to be mapped.'
    'Can be used to make an imported dataset compatible with the database.')
  parser.set_defaults(func=renameObjects)
  parser.add_argument('--names_dict', required=True,
    help='Map from old names to new names. E.g. \'{"Car": "car", "Truck": "car"}\'')
  parser.add_argument('--where_object', default='TRUE',
    help='SQL "where" clause for the "objects" table.')
  parser.add_argument('--discard_other_objects', action='store_true',
    help='Discard objects with names not in the keys or the values of "names_dict".')

def renameObjects (c, args):
  namesmap = literal_eval(args.names_dict)

  # Maybe delete other objects.
  if args.discard_other_objects:
    c.execute('SELECT objectid FROM objects WHERE name NOT IN (%s) AND (%s)' %
      (namesmap.keys() + namesmap.values(), args.where_object))
    other_objectids = c.fetchall()
    logging.info('Will delete %d objects with names not in the map.')
    for objectid, in other_objectids:
      deleteObject(c, objectid)

  # Remap the rest.
  for key, value in namesmap.items():
    c.execute('UPDATE objects SET name=? WHERE name=? AND (%s)' % args.where_object, (value, key))



def resizeAnnotationsParser(subparsers):
  parser = subparsers.add_parser('resizeAnnotations',
    description='Resize all information about images and objects. '
    'Use when images were scaled and the annotations need to be updated. '
    'This command does not scale images themselves. '
    'The "old" image size is obtrained from "images" table.')
  parser.set_defaults(func=resizeAnnotations)
  parser.add_argument('--where_image', default='TRUE',
    help='the SQL "where" clause for "images" table. '
    'E.g. to change imagefile of JPG pictures from directory "from/mydir" only, use: '
    '\'imagefile LIKE "from/mydir/%%"\'')
  parser.add_argument('--target_width', type=int, required=False,
    help='The width each image was scaled to.')
  parser.add_argument('--target_height', type=int, required=False,
    help='The height each image was scaled to.')

def resizeAnnotations(c, args):
  
  if args.target_width is None and args.target_height is None:
    raise ValueError('One or both "target_width", "target_height" should be specified.')

  # Process image by image.
  c.execute('SELECT imagefile,width,height FROM images WHERE (%s)' % args.where_image)
  for imagefile, old_width, old_height in progressbar(c.fetchall()):

    # Figure out scaling, depending on which of target_width and target_height is given.
    if args.target_width is not None and args.target_height is not None:
      percent_x = args.target_width / float(old_width)
      percent_y = args.target_height / float(old_height)
      target_width = args.target_width
      target_height = args.target_height
    if args.target_width is None and args.target_height is not None:
      percent_y = args.target_height / float(old_height)
      percent_x = percent_y
      target_width = int(old_width * percent_x)
      target_height = args.target_height
    if args.target_width is not None and args.target_height is None:
      percent_x = args.target_width / float(old_width)
      percent_y = percent_x
      target_width = args.target_width
      target_height = int(old_height * percent_y)
    logging.info('Scaling "%s" with percent_x=%.2f, percent_y=%.2f' % 
        (imagefile, percent_x, percent_y))

    # Update images.
    c.execute('UPDATE images SET width=?,height=? WHERE imagefile=?',
        (target_width, target_height, imagefile))

    # Update objects.
    c.execute('SELECT objectid,x1,y1,width,height FROM objects WHERE imagefile=?', (imagefile,))
    for objectid,x1,y1,width,height in c.fetchall():
      if x1 is not None:
        x1 = int(x1 * percent_x)
      if y1 is not None:
        y1 = int(y1 * percent_y)
      if width is not None:
        width = int(width * percent_x)
      if height is not None:
        height = int(height * percent_y)
      c.execute('UPDATE objects SET x1=?,y1=?,width=?,height=? WHERE objectid=?',
        (x1,y1,width,height,objectid))

    # Update polygons.
    c.execute('SELECT id,x,y FROM polygons p INNER JOIN objects o ON o.objectid=p.objectid WHERE imagefile=?', (imagefile,))
    for id_,x,y in c.fetchall():
      x = int(x * percent_x)
      y = int(y * percent_y)
      c.execute('UPDATE polygons SET x=?,y=? WHERE id=?', (x,y,id_))



def propertyToNameParser(subparsers):
  parser = subparsers.add_parser('propertyToName',
    description='Assigns names to propert')
  parser.set_defaults(func=propertyToName)
  parser.add_argument('--property', required=True,
    help='Property name to assign to objects.')
  parser.add_argument('--assign_null', action='store_true',
    help='When specifies, the name is assigned null '
    'when property is absent for that object.')
  parser.add_argument('--delete_property_after', action='store_true',
    help='When specifies, deletes that property when done')

def propertyToName(c, args):
  if args.assign_null:
    c.execute('UPDATE objects SET name = ('
        'SELECT value FROM properties '
        'WHERE objects.objectid = properties.objectid AND key=?'
      ')', (args.property,)
    )
  else:
    c.execute('UPDATE objects SET name = ('
        'SELECT value FROM properties '
        'WHERE objects.objectid = properties.objectid AND key=?'
      ') WHERE objectid IN ('
        'SELECT objectid FROM properties WHERE key=?'
      ')', (args.property, args.property)
    )

  if args.delete_property_after:
    c.execute('DELETE FROM properties WHERE key=?', (args.property,))
  

