import os, sys, os.path as op
import numpy as np
import cv2
import logging
import sqlite3
import imageio
from math import ceil
from glob import glob
from pprint import pformat
from datetime import datetime
from progressbar import progressbar
from ast import literal_eval

from .backendDb import makeTimeString, deleteImage, objectField, createDb
from .backendImages import getPictureSize, ImageryReader, VideoWriter, PictureWriter
from .util import drawScoredRoi, roi2bbox
from .utilExpandBoxes import expandRoiToRatio, expandRoi


def add_parsers(subparsers):
  addVideoParser(subparsers)
  addPicturesParser(subparsers)
  headImagesParser(subparsers)
  tailImagesParser(subparsers)
  moduloAnglesParser(subparsers)
  expandBoxesParser(subparsers)
  moveDirParser(subparsers)
  addDbParser(subparsers)
  splitDbParser(subparsers)
  mergeObjectDuplicatesParser(subparsers)
  renameObjectsParser(subparsers)
  polygonsToMaskParser(subparsers)


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


def moduloAnglesParser(subparsers):
  parser = subparsers.add_parser('moduloAngles',
    description='Update all value of specified property to "value % 360"')
  parser.add_argument('--property', required=True,
    help='The key field with angle property of the "properties" table.')
  parser.set_defaults(func=moduloAngles)

def moduloAngles(c, args):
  c.execute('UPDATE properties SET value = CAST(value as decimal) %% 360.0 WHERE key="%s"' % args.property)


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
    help='Until <Esc> key, display how each object intersects others. '
    'Bad objects are shown as as red, good as blue, while the others as green.')

def expandBoxes (c, args):
  if args.with_display:
    imreader = ImageryReader(rootdir=args.rootdir)

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
        roi = expandRoiToRatio(roi, args.expand_perc, args.target_ratio)
      else:
        roi = expandRoi(roi, (args.expand_perc, args.expand_perc))
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


def moveDirParser(subparsers):
  parser = subparsers.add_parser('moveDir',
    description='Change imagefile and maskfile.')
  parser.set_defaults(func=moveDir)
  parser.add_argument('--where_image', default='TRUE',
    help='the SQL "where" clause for "images" table. '
    'E.g. to change imagefile of JPG pictures from directory "from/mydir" only, use: '
    '\'imagefile LIKE "from/mydir/%%"\'')
  parser.add_argument('--image_dir')
  parser.add_argument('--mask_dir')

def moveDir (c, args):

  if args.image_dir:
    logging.debug ('Moving image dir to: %s' % args.image_dir)
    c.execute('SELECT imagefile FROM images WHERE (%s)' % args.where_image)
    imagefiles = c.fetchall()

    for oldfile, in progressbar(imagefiles):
      newfile = op.join (args.image_dir, op.basename(oldfile))
      c.execute('UPDATE images SET imagefile=? WHERE imagefile=?', (newfile, oldfile))
      c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?', (newfile, oldfile))

  if args.mask_dir:
    logging.debug ('Moving mask dir to: %s' % args.mask_dir)
    c.execute('SELECT maskfile FROM images WHERE (%s)' % args.where_image)
    maskfiles = c.fetchall()

    for oldfile, in progressbar(maskfiles):
      if oldfile is not None:
        newfile = op.join (args.mask_dir, op.basename(oldfile))
        c.execute('UPDATE images SET maskfile=? WHERE maskfile=?', (newfile, oldfile))


def addDbParser(subparsers):
  parser = subparsers.add_parser('addDb',
    description='Adds info from "db_file" to the current open database. Objects can be merged.'
    'Duplicate "imagefile" entries are ignore, but all associated objects are added.')
  parser.set_defaults(func=addDb)
  parser.add_argument('--db_file', required=True)
  parser.add_argument('--merge_duplicate_objects', action='store_true')
    
def addDb (c, args):
  conn_add = sqlite3.connect('file:%s?mode=ro' % args.db_file, uri=True)
  c_add = conn_add.cursor()

  # Copy images.
  c_add.execute('SELECT * FROM images')
  logging.info('Copying images.')
  for image_entry in progressbar(c_add.fetchall()):
    c.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?)', image_entry)

  # Find the max "match" in "matches" table. New matches will go after that value.
  c.execute('SELECT MAX(match) FROM matches')
  max_match = c.fetchone()[0]
  
  # Copy all other tables.
  c_add.execute('SELECT * FROM objects')
  logging.info('Copying objects.')
  for object_entry_add in progressbar(c_add.fetchall()):
    objectid_add = objectField(object_entry_add, 'objectid')

    # Copy objects.
    c.execute('INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
      'VALUES (?,?,?,?,?,?,?)', object_entry_add[1:])
    objectid = c.lastrowid

    # Copy properties.
    c_add.execute('SELECT key,value FROM properties WHERE objectid=?', (objectid_add,))
    for key, value in c_add.fetchall():
      c.execute('INSERT INTO properties(objectid,key,value) VALUES (?,?,?)', (objectid,key,value))

    # Copy polygons.
    c_add.execute('SELECT x,y,name FROM polygons WHERE objectid=?', (objectid_add,))
    for x, y, name in c_add.fetchall():
      c.execute('INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)', (objectid,x,y,name))

    # Copy matches.
    c_add.execute('SELECT match FROM matches WHERE objectid=?', (objectid_add,))
    for match, in c_add.fetchall():
      c.execute('INSERT INTO matches(objectid,match) VALUES (?,?)', (objectid, match + max_match))

  conn_add.close()



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
    random.shuffle(imagefiles)

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



def mergeObjectDuplicatesParser(subparsers):
  parser = subparsers.add_parser('mergeObjectDuplicates',
    description='Merge objects with identical fields imagefile,x1,y1,width,height.')
  parser.set_defaults(func=mergeObjectDuplicates)
    
def mergeObjectDuplicates (c, args):
  c.execute('SELECT imagefile FROM images')
  for imagefile, in progressbar(c.fetchall()):
    logging.debug('Image %s' % imagefile)

    # Duplicates are defined by fields x1,y1,width,height.
    c.execute('SELECT x1,y1,width,height,COUNT(1) FROM objects WHERE imagefile=? '
            'GROUP BY x1,y1,width,height HAVING COUNT(1) > 1', (imagefile,))
    duplicates = c.fetchall()
    logging.info('Found %d duplicates in imagefile "%s"' % (len(duplicates), imagefile))

    for x1, y1, width, height, num in duplicates:
      logging.debug('Duplicate (x1,y1,width,height): %d,%d,%d,%d is in %d objects' %
        (x1, y1, width, height, num))

      # Get all object entries for the duplicates.
      c.execute('SELECT objectid FROM objects WHERE imagefile=? AND x1=? AND y1=? AND width=? AND height=?',
        (imagefile, x1, y1, width, height))
      objectids = [x for x, in c.fetchall()]
      if len(objectids) == 1:  # No duplicates.
        continue
      objectids_list = (','.join([str(x) for x in objectids]))
      objectids_list_but1 = (','.join([str(x) for x in objectids[1:]]))

      # Merge score.
      c.execute('SELECT DISTINCT(score) FROM objects WHERE objectid IN (%s) AND score IS NOT NULL' % objectids_list)
      scores = [x for x, in c.fetchall()]
      logging.debug('Merged objects %s have scores %s' % (objectids_list, scores))
      if len(scores) > 1:
        raise ValueError('Too many distinct "score" (%s) for objectids (%s)' % (str(scores), objectids_list))
      logging.debug('Writing score=%f to objectid=%s.' % (scores[0], objectids[0]))
      c.execute('UPDATE objects SET score=? WHERE objectid=?;', (scores[0], objectids[0]))

      # Merge name.
      c.execute('SELECT DISTINCT(name) FROM objects WHERE objectid IN (%s) AND name IS NOT NULL' % objectids_list)
      names = [x for x, in c.fetchall()]
      logging.debug('Merged objects %s have names %s' % (objectids_list, names))
      if len(names) > 1:
        raise ValueError('Too many distinct "name" (%s) for objectids (%s)' % (str(names), objectids_list))
      logging.debug('Writing name=%s to objectid=%s.' % (names[0], objectids[0]))
      c.execute('UPDATE objects SET name=? WHERE objectid=?;', (names[0], objectids[0]))

      # Delete all duplicate entries of "objects" table.
      c.execute('DELETE FROM objects WHERE objectid IN (%s);' % objectids_list_but1)

      # Merge properties.
      for objectid in objectids[1:]:
        c.execute('UPDATE properties SET objectid=? WHERE objectid=?;', (objectids[0], objectid))
      c.execute('SELECT key,COUNT(DISTINCT(value)) FROM properties WHERE objectid=? GROUP BY key', (objectids[0],))
      for key, num in c.fetchall():
        if num > 1:
          raise ValueError('%d duplicate values for key=%s for objectid=%s' % (num, key, objectids[0]))

      # Merge polygons.
      c.execute('SELECT COUNT(1) FROM polygons WHERE objectid IN (%s) GROUP BY objectid;' % objectids_list)
      num = c.fetchone()
      num = num[0] if num is not None else 0
      logging.debug('Found %d polygons for objectids %s' % (num, objectids_list))
      if num > 1:
        c.execute('UPDATE polygons SET objectid=?, name=(ifnull(name, "")  || objectid) WHERE objectid IN (%s);' % 
          objectids_list, (objectids[0],))
      if num >= 1:
        c.execute('UPDATE polygons SET objectid=? WHERE objectid IN (%s);' % objectids_list, (objectids[0],))

      # Matches not implemented yet, because there were not tasks for them so far.
      c.execute('SELECT COUNT(1) FROM matches WHERE objectid IN (%s)' % objectids_list_but1)
      if c.fetchone()[0] > 0:
        raise NotImplementedError('Marging matches table not implemented yet.')



def renameObjectsParser(subparsers):
  parser = subparsers.add_parser('renameObjects',
    description='Map object names. '
    'Can be used to make an imported dataset compatible with the database.')
  parser.set_defaults(func=renameObjects)
  parser.add_argument('--names_dict', required=True,
    help='Map from old names to new names. E.g. {"Car": "car", "Truck": "car"}')
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



def polygonsToMaskParser(subparsers):
  parser = subparsers.add_parser('polygonsToMask',
    description='Convert polygons of an object into a mask, and write it as maskfile.'
    'If there are polygon entries with different names, consider them as different polygons. '
    'Masks from each of these polygons are summed up and normalized to their number. '
    'The result is a black-and-white mask when there is only one polygon, and '
    'a grayscale mask when there are multiple polygons.')
  parser.set_defaults(func=polygonsToMask)
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--mask_pictures_dir',
    help='the directory where to write mask pictures to.')
  group.add_argument('--mask_video_file',
    help='the video file where to write masks to.')
  parser.add_argument('--overwrite', action='store_true',
    help='overwrite images or video.')
  parser.add_argument('--skip_empty_masks', action='store_true',
    help='do not write black masks with no objects.')

def polygonsToMask (c, args):

  # Create mask writer.
  if args.mask_video_file:
    imwriter = VideoWriter(rootdir=args.rootdir, vmaskfile=args.mask_video_file, overwrite=args.overwrite)
  elif args.mask_pictures_dir:
    imwriter = PictureWriter(rootdir=args.rootdir)
  else:
    raise ValueError('Specify either "mask_video_file" or "mask_pictures_dir".')

  # Iterate images.
  c.execute('SELECT imagefile,width,height FROM images')
  for imagefile, width, height in progressbar(c.fetchall()):
    mask_per_image = np.zeros((height, width), dtype=np.int32)

    # Iterate objects.
    c.execute('SELECT objectid FROM objects WHERE imagefile=?', (imagefile,))
    for objectid, in c.fetchall():
      mask_per_object = np.zeros((height, width), dtype=np.int32)

      # Iterate multiple polygons (if any) of the object.
      c.execute('SELECT DISTINCT(name) FROM polygons WHERE objectid=?', (objectid,))
      polygon_names = c.fetchall()
      for polygon_name, in polygon_names:

        # Draw a polygon.
        if polygon_name is None:
          c.execute('SELECT x,y FROM polygons WHERE objectid=?', (objectid,))
        else:
          c.execute('SELECT x,y FROM polygons WHERE objectid=? AND name=?', (objectid, polygon_name))
        pts = [[pt[0], pt[1]] for pt in c.fetchall()]
        logging.debug('Polygon "%s" of object %d consists of points: %s' % (polygon_name, objectid, str(pts)))
        mask_per_polygon = np.zeros((height, width), dtype=np.int32)
        cv2.fillConvexPoly(mask_per_polygon, np.asarray(pts, dtype=np.int32), 255)
        mask_per_object += mask_per_polygon

      # Area inside all polygons is white, outside all polygons is black, else gray.
      if len(polygon_names) > 1:
        mask_per_object = mask_per_object // len(polygon_names)

      mask_per_image += mask_per_object
    mask_per_image = mask_per_image.astype(np.uint8)

    # Maybe skip empty mask.
    if np.sum(mask_per_image) == 0 and args.skip_empty_masks:
      continue

    if args.mask_video_file:
      maskfile = imwriter.maskwrite(mask_per_image)
    elif args.mask_pictures_dir:
      maskname = '%s.png' % op.splitext(op.basename(imagefile))[0]
      maskfile = op.join(args.out_pictures_dir, maskname)
      imwriter.maskwrite(maskfile, mask_per_image)
    c.execute('UPDATE images SET maskfile=? WHERE imagefile=?', (maskfile, imagefile))

  imwriter.close()
