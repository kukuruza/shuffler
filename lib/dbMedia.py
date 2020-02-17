import os, sys, os.path as op
import numpy as np
import cv2
import logging
from progressbar import progressbar
from pprint import pformat
from ast import literal_eval
from datetime import timedelta
from math import sqrt

from .backendDb import imageField, objectField, polygonField, deleteImage, parseTimeString, makeTimeString
from .backendMedia import MediaReader, MediaWriter
from .util import drawTextOnImage, drawMaskOnImage, drawMaskAside, drawScoredPolygon, drawScoredRoi, bbox2roi, applyLabelMappingToMask
from .utilExpandBoxes import cropPatch


def add_parsers(subparsers):
  writeMediaParser(subparsers)
  cropMediaParser(subparsers)
  cropObjectsParser(subparsers)
  polygonsToMaskParser(subparsers)
  writeMediaGridByTimeParser(subparsers)
  repaintMaskParser(subparsers)



def cropMediaParser(subparsers):
  parser = subparsers.add_parser('cropMedia',
    description='Crops images to a single ROI.')
  parser.set_defaults(func=cropMedia)
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--out_rootdir',
    help='Specify, if rootdir changed for the output imagery.')
  parser.add_argument('--image_path', required=True,
    help='the directory for pictures OR video file, where to write mask crop pictures.')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file, where to write mask crop pictures.')
  parser.add_argument('--x1', type=int, required=True)
  parser.add_argument('--y1', type=int, required=True)
  parser.add_argument('--x2', type=int, required=True)
  parser.add_argument('--y2', type=int, required=True)
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')

def cropMedia(c, args):
  imreader = MediaReader(rootdir=args.rootdir)
  # Create a writer. Rootdir may be changed.
  out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
  imwriter = MediaWriter(rootdir=out_rootdir, media_type=args.media,
    image_media=args.image_path, mask_media=args.mask_path, overwrite=args.overwrite)

  target_width  = args.x2 - args.x1
  target_height = args.y2 - args.y1

  c.execute('SELECT * FROM images')
  for image_entry in progressbar(c.fetchall()):
    imagefile = imageField(image_entry, 'imagefile')
    maskfile = imageField(image_entry, 'maskfile')

    image = imreader.imread(imagefile)
    image = image[args.y1:args.y2, args.x1:args.x2, :]
    if args.mask_path is not None and maskfile is not None:
      mask = imreader.maskread(maskfile)
      mask = mask[args.y1:args.y2, args.x1:args.x2]

    # Write an image.
    if args.image_path is not None:
      imagefile_new = imwriter.imwrite(image)
    else:
      imagefile_new = None

    # Write mask.
    if args.mask_path is not None and maskfile is not None:
      maskfile_new = imwriter.maskwrite(mask)
    else:
      maskfile_new = None

    # Update the database entry.
    if maskfile_new is not None:
      c.execute('UPDATE images SET maskfile=?,width=?,height=? WHERE imagefile=?', (maskfile_new,target_width,target_height,imagefile))
    if imagefile_new is not None:
      c.execute('UPDATE images SET imagefile=?,width=?,height=? WHERE imagefile=?', (imagefile_new,target_width,target_height,imagefile))
      c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?', (imagefile_new, imagefile))

  c.execute('UPDATE objects SET x1=(SELECT x1 FROM objects)-?, y1=(SELECT y1 FROM objects)-?', (args.x1, args.y1))
  # TODO: update polygons.
  imwriter.close()



def cropObjectsParser(subparsers):
  parser = subparsers.add_parser('cropObjects',
    description='Crops object patches to pictures or video and saves their info as a db. '
    'All imagefiles and maskfiles in the db will be replaced with the crops, one per object.')
  parser.set_defaults(func=cropObjects)
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--image_path', required=True,
    help='the directory for pictures OR video file, where to write mask crop pictures.')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file, where to write mask crop pictures.')
  parser.add_argument('--where_object', default='TRUE',
    help='SQL "where" clause for the "objects" table.')
  parser.add_argument('--target_width', required=False, type=int)
  parser.add_argument('--target_height', required=False, type=int)
  parser.add_argument('--edges', default='distort',
    choices={'distort', 'constant', 'background', 'variable_size'},
    help='''"distort" distorts the patch to get to the desired ratio,
            "constant" keeps the ratio but pads the patch with zeros,
            "background" keeps the ratio but includes image background.
            "variable_size" does not change the crop dimensions. Target width and height are ignored.''')
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')

def cropObjects(c, args):
  imreader = MediaReader(rootdir=args.rootdir)
  imwriter = MediaWriter(media_type=args.media, rootdir=args.rootdir,
    image_media=args.image_path, mask_media=args.mask_path, overwrite=args.overwrite)

  c.execute('SELECT o.objectid,o.imagefile,o.x1,o.y1,o.width,o.height,o.name,o.score,i.maskfile,i.timestamp '
    'FROM objects AS o INNER JOIN images AS i ON i.imagefile = o.imagefile WHERE (%s) ORDER BY objectid' %
    args.where_object)
  entries = c.fetchall()
  logging.debug(pformat(entries))

  c.execute('DELETE FROM images')

  # If only one object category was sampled, other objects should be preserved.
  if args.where_object:
    # Will later erase all up to this id.
    c.execute('SELECT MAX(objectid) FROM objects')
    old_row_id = c.fetchone()[0]
    last_row_id = old_row_id

  for objectid,imagefile,x1,y1,width,height,name,score,maskfile,timestamp in progressbar(entries):
    logging.debug ('Processing object %d from imagefile %s.' % (objectid, imagefile))
    roi = bbox2roi((x1,y1,width,height))

    # Write image.
    image = imreader.imread(imagefile)
    logging.debug('Cropping roi=%s from image of shape %s' % (roi, image.shape))
    image = cropPatch(image, roi, args.target_height, args.target_width, args.edges)
    imagefile_cropped = imwriter.imwrite(image, namehint='%06d' % objectid)

    # Write mask.
    if args.mask_path is not None and maskfile is not None:
      mask = imreader.maskread(maskfile)
      mask = cropPatch(mask, roi, args.target_height, args.target_width, args.edges)
      maskfile_cropped = imwriter.maskwrite(mask, namehint='%06d' % objectid)
    elif args.mask_path is None:
      maskfile_cropped = None

    logging.debug('Recording imagefile %s and maskfile %s.' % (imagefile_cropped, maskfile_cropped))
    c.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',
      (imagefile_cropped, args.target_width, args.target_height, maskfile_cropped, timestamp, name, score))

    # If only one object category was sampled, other objects should be preserved.
    if args.where_object:
      c.execute('SELECT * FROM objects WHERE imagefile=? AND x1<? AND y1<? AND x1+width>? AND y1+height>?',
        (imagefile, x1 + width, y1 + height, x1, y1))
      objects = c.fetchall()

      for iobj,object_ in enumerate(objects):
        objectid_obj = objectField(object_, 'objectid')
        x1_obj = objectField(object_, 'x1')
        y1_obj = objectField(object_, 'y1')
        width_obj = objectField(object_, 'width')
        height_obj = objectField(object_, 'height')
        name_obj = objectField(object_, 'name')
        score_obj = objectField(object_, 'score')
        new_id = last_row_id + iobj + 1
        logging.debug('Copying objectid=%d (%s) to new objectid=%d ', objectid_obj, name_obj, new_id)
        c.execute('INSERT INTO objects(objectid,imagefile,x1,y1,width,height,name,score) VALUES (?,?,?,?,?,?,?,?)',
          (new_id, imagefile_cropped, x1_obj-x1, y1_obj-y1, width_obj, height_obj, name_obj, score_obj))     
 
      # Will be used for the next crop.
      last_row_id += len(objects)
      
  if args.where_object:
    logging.debug('Removing all objects up to %d', old_row_id)
    c.execute('DELETE FROM objects WHERE objectid <= ?', (old_row_id,))
  else:
    # Simple case of only one object in the crop -- the one which was cropped.
    c.execute('UPDATE objects SET x1=0,y1=0,width=?,height=?', (args.target_width, args.target_height))

  # TODO: copy properties, polygons, and matches.
  c.execute('DELETE FROM properties')
  c.execute('DELETE FROM polygons')
  c.execute('DELETE FROM matches')

  imwriter.close()



def writeMediaParser(subparsers):
  parser = subparsers.add_parser('writeMedia',
    description='Export images as a directory with pictures or as a video, '
    'and change the database imagefiles and maskfiles to match the recordings.')
  parser.set_defaults(func=writeMedia)
  parser.add_argument('--out_rootdir',
    help='Specify, if rootdir changed for the output imagery.')
  parser.add_argument('--where_image', default='TRUE',
    help='SQL where clause for the "images" table.')
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--image_path',
    help='the directory for pictures OR video file, where to write masks.')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file, where to write masks.')
  parser.add_argument('--mask_mapping_dict', 
    help='how values in maskfile are drawn. E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--mask_aside', action='store_true',
    help='Image and mask side by side.')
  group.add_argument('--mask_alpha', type=float,
    help='transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--with_imageid', action='store_true', help='print frame number.')
  parser.add_argument('--with_objects', action='store_true', help='draw objects on top.')
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')

def writeMedia (c, args):
  imreader = MediaReader(rootdir=args.rootdir)

  # For overlaying masks.
  labelmap = literal_eval(args.mask_mapping_dict) if args.mask_mapping_dict else None
  logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

  # Create a writer. Rootdir may be changed.
  out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
  imwriter = MediaWriter(rootdir=out_rootdir, media_type=args.media,
    image_media=args.image_path, mask_media=args.mask_path, overwrite=args.overwrite)

  c.execute('SELECT imagefile,maskfile FROM images WHERE %s' % args.where_image)
  entries = c.fetchall()

  logging.info('Deleting entries which are not in "where_image".')
  c.execute('SELECT imagefile FROM images WHERE NOT %s' % args.where_image)
  for imagefile, in c.fetchall():
    deleteImage(c, imagefile)

  logging.info('Writing imagery and updating the database.')
  for imagefile, maskfile in progressbar(entries):

    logging.debug ('Imagefile "%s"' % imagefile)
    if args.image_path is not None:
      image = imreader.imread(imagefile)

    # Overlay the mask.
    if maskfile is not None:
      mask = imreader.maskread(maskfile)
      if args.image_path is not None:
        if args.mask_aside:
          image = drawMaskAside(image, mask, labelmap=labelmap)
        elif args.mask_alpha is not None:
          image = drawMaskOnImage(image, mask, alpha=args.mask_alpha, labelmap=labelmap)
    else:
      mask = None
      logging.debug('No mask for this image.')

    # Overlay imagefile.
    if args.with_imageid:
      drawTextOnImage(image, op.basename(imagefile))

    # Draw objects as polygons (preferred) or ROI.
    if args.with_objects:
      c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
      object_entries = c.fetchall()
      logging.debug('Found %d objects' % len(object_entries))
      for object_entry in object_entries:
        objectid = objectField(object_entry, 'objectid')
        roi   = objectField(object_entry, 'roi')
        score = objectField(object_entry, 'score')
        name  = objectField(object_entry, 'name')
        c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid,))
        polygon_entries = c.fetchall()
        if len(polygon_entries) > 0:
          logging.debug('showing object with a polygon.')
          polygon = [(polygonField(p, 'x'), polygonField(p, 'y')) for p in polygon_entries]
          drawScoredPolygon (image, polygon, label=name, score=score)
        elif roi is not None:
          logging.debug('showing object with a bounding box.')
          drawScoredRoi (image, roi, label=name, score=score)
        else:
          raise Exception('Neither polygon, nor bbox is available for objectid %d' % objectid)

    # Write an image.
    if args.image_path is not None:
      imagefile_new = imwriter.imwrite(image)
    else:
      imagefile_new = None

    # Write mask.
    if args.mask_path is not None and maskfile is not None:
      maskfile_new = imwriter.maskwrite(mask)
    else:
      maskfile_new = None

    # Update the database entry.
    if maskfile_new is not None:
      c.execute('UPDATE images SET maskfile=? WHERE imagefile=?', (maskfile_new,imagefile))
    if imagefile_new is not None:
      c.execute('UPDATE images SET imagefile=? WHERE imagefile=?', (imagefile_new,imagefile))
      c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?', (imagefile_new,imagefile))

  imwriter.close()


def polygonsToMaskParser(subparsers):
  parser = subparsers.add_parser('polygonsToMask',
    description='Convert polygons of an object into a mask, and write it as maskfile.'
    'If there are polygon entries with different names, consider them as different polygons. '
    'Masks from each of these polygons are summed up and normalized to their number. '
    'The result is a black-and-white mask when there is only one polygon, and '
    'a grayscale mask when there are multiple polygons.')
  parser.set_defaults(func=polygonsToMask)
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file, where to write masks.')
  parser.add_argument('--overwrite', action='store_true',
    help='overwrite images or video.')
  parser.add_argument('--skip_empty_masks', action='store_true',
    help='do not write black masks with no objects.')
  parser.add_argument('--substitute_with_box', action='store_true',
    help='if a polygon is not available, allow to use the bounding box.')

def polygonsToMask (c, args):
  imwriter = MediaWriter(media_type=args.media, rootdir=args.rootdir,
    mask_media=args.mask_path, overwrite=args.overwrite)

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
        cv2.fillPoly(mask_per_polygon, [np.asarray(pts, dtype=np.int32)], 255)
        mask_per_object += mask_per_polygon

      # Area inside all polygons is white, outside all polygons is black, else gray.
      if len(polygon_names) > 1:
        mask_per_object = mask_per_object // len(polygon_names)

      # If there are no polygons, maybe substitute with roi.
      elif len(polygon_names) == 0 and args.substitute_with_box:
        c.execute('SELECT * FROM objects WHERE objectid=?', (objectid,))
        object_entry = c.fetchone()
        roi = objectField(object_entry, 'roi')
        cv2.rectangle(mask_per_object, (roi[1],roi[0]), (roi[3],roi[2]), 255, thickness=-1)

      mask_per_image += mask_per_object
    mask_per_image = np.minimum(mask_per_image, 255)  # Objects overlay on each other.
    mask_per_image = mask_per_image.astype(np.uint8)

    # Maybe skip empty mask.
    if np.sum(mask_per_image) == 0 and args.skip_empty_masks:
      continue

    out_maskfile = imwriter.maskwrite(mask_per_image) #, namehint='%s.png' % op.splitext(imagefile)[0])
    c.execute('UPDATE images SET maskfile=? WHERE imagefile=?', (out_maskfile, imagefile))

  imwriter.close()


def writeMediaGridByTimeParser(subparsers):
  parser = subparsers.add_parser('writeMediaGridByTime',
    description='Export images, arranged in a grid, as a directory with pictures or as a video. '
    'Grid arranged by directory of imagefile (imagedir), and can be set manually.')
  parser.set_defaults(func=writeMediaGridByTime)
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--image_path', required=True,
    help='the directory for pictures OR video file, where to write mask crop pictures.')
  parser.add_argument('--winwidth', type=int, default=360,
    help='target output width of each video in a grid.')
  parser.add_argument('--fps', type=int, default=5)
  parser.add_argument('--gridY', type=int,
    help='if specified, use the grid of "gridY" cells wide. Infer gridX.')
  parser.add_argument('--imagedirs', nargs='+',
    help='if specified, use these imagedirs instead of inferring from imagefile.')
  parser.add_argument('--num_seconds', type=int,
    help='If specified, stop there.')
  parser.add_argument('--with_timestamp', action='store_true',
    help='Draw time on top.')
  parser.add_argument('--overwrite', action='store_true',
    help='overwrite video if it exists.')

def writeMediaGridByTime (c, args):
  imreader = MediaReader(rootdir=args.rootdir)
  imwriter = MediaWriter(media_type=args.media,
    image_media=args.image_path, mask_media=args.mask_path, overwrite=args.overwrite)

  if not args.imagedirs:
    c.execute("SELECT DISTINCT(rtrim(imagefile, replace(imagefile, '/', ''))) FROM images")
    imagedirs = c.fetchall()
    imagedirs = [x.strip('/') for x, in imagedirs]
  else:
    imagedirs = args.imagedirs

  logging.info('Found %d distinct image directories/videos:\n%s' %
    (len(imagedirs), pformat(imagedirs)))
  num_cells = len(imagedirs)
  gridY = int(sqrt(num_cells)) if args.gridY is None else args.gridY
  gridX = num_cells // gridY
  logging.info('Will use grid %d x %d' % (gridY, gridX))
  
  # Currently will use the first image to find the target height from args.winwidth.
  # "width" and "height" are the target dimensions.
  c.execute('SELECT width,height FROM images')
  width, height = c.fetchone()
  height = height * args.winwidth // width
  width = args.winwidth

  c.execute('SELECT imagefile,timestamp FROM images')
  image_entries = c.fetchall()
  image_entries = [(imagefile, parseTimeString(timestamp)) for imagefile, timestamp in image_entries]
  image_entries = sorted(image_entries, key=lambda x: x[1])

  time_min = min(image_entries, key=lambda x: x[1])[1]
  time_max = max(image_entries, key=lambda x: x[1])[1]
  logging.info ('Min time: "%s", max time: "%s".' % (time_min, time_max))
  delta_seconds = (time_max - time_min).total_seconds()
  # User may limit the time to write.
  if args.num_seconds is not None:
    delta_seconds = min(delta_seconds, args.num_seconds)

  ientry = 0
  seconds_offsets = np.arange(0, int(delta_seconds), 1.0 / args.fps).tolist()

  for seconds_offset in progressbar(seconds_offsets):

    out_time = time_min + timedelta(seconds=seconds_offset)
    logging.debug('Out-time=%s, in-time=%s' % (out_time, image_entries[ientry][1]))

    # For all the entries that happened before the frame.
    while image_entries[ientry][1] < out_time:

      imagefile, in_time = image_entries[ientry]
      logging.debug('Image entry %d.' % ientry)
      ientry += 1
      # Skip those of no interest.
      if op.dirname(imagefile) not in imagedirs:
        logging.debug('Image dir %s not in %s' % (op.dirname(imagefile), imagedirs))
        continue
      gridid = imagedirs.index(op.dirname(imagefile))

      # Read and scale.
      image = imreader.imread(imagefile)
      image = cv2.resize(image, dsize=(width,height))
      assert len(image.shape) == 3  # Now only color images.
      if args.with_timestamp:
        drawTextOnImage(image, makeTimeString(in_time))
      #drawTextOnImage(image, op.basename(op.dirname(imagefile)))

      # Lazy initialization.
      if 'grid' not in locals():
        grid = np.zeros((height * gridY, width * gridX, 3), dtype=np.uint8)

      logging.debug('Writing %s into gridid %d' % (imagefile, gridid))
      grid[height * (gridid // gridX) : height * (gridid // gridX + 1),
           width  * (gridid % gridX)  : width  * (gridid % gridX + 1),
           :] = image.copy()
      
      imwriter.imwrite(grid)

  imwriter.close()



def repaintMaskParser(subparsers):
  parser = subparsers.add_parser('repaintMask',
    description='Repaint specific colors in mask into different colors.')
  parser.set_defaults(func=repaintMask)
  parser.add_argument('--media', choices=['pictures', 'video'], required=True,
    help='output either a directory with pictures or a video file.')
  parser.add_argument('--mask_path',
    help='the directory for pictures OR video file, where to write masks. '
    'Can not overwrite the same mask dir / video.')
  parser.add_argument('--mask_mapping_dict', required=True,
    help='values mapping for repainting. E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
  parser.add_argument('--overwrite', action='store_true',
    help='overwrite images or video.')
  parser.add_argument('--out_rootdir',
    help='Specify, if rootdir changed for the output imagery.')

def repaintMask (c, args):
  labelmap = literal_eval(args.mask_mapping_dict)
  logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

  imreader = MediaReader(rootdir=args.rootdir)

  out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
  imwriter = MediaWriter(rootdir=out_rootdir,
    media_type=args.media, mask_media=args.mask_path, overwrite=args.overwrite)

  # Iterate images.
  c.execute('SELECT maskfile FROM images')
  maskfiles = c.fetchall()

  # Find out if all the masks are in the same directory.
  # If so, use a namehint, that is, new maskfiles will have the same name as original.
  # Otherwise, new maskfiles will be named seqentially from 0.
  # Regardless, namehint is currently only used when media=="pictures".
  maskdirs = set([op.dirname(maskfile) for maskfile in maskfiles])
  use_namehint = len(maskdirs) == 1

  for maskfile, in progressbar(maskfiles):
    if maskfile is not None:
      # Read mask.
      mask = imreader.maskread(maskfile)
      # Repaint mask.
      mask = applyLabelMappingToMask(mask, labelmap).astype(np.uint8)
      # Write mask to video and to the db.
      maskfile_new = imwriter.maskwrite(mask, namehint=(maskfile if use_namehint else None))
      c.execute('UPDATE images SET maskfile=? WHERE maskfile=?', (maskfile_new,maskfile))

  imwriter.close()
