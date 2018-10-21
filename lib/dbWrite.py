import os, sys, os.path as op
import numpy as np
import cv2
import logging
from progressbar import progressbar
from pprint import pformat
from ast import literal_eval

from .backendDb import objectField, polygonField
from .backendImages import ImageryReader, VideoWriter, PictureWriter
from .util import drawImageId, drawMaskOnImage, cropPatch, bbox2roi


def add_parsers(subparsers):
  writeImagesParser(subparsers)
  cropObjectsParser(subparsers)


# class DatasetWriter:
#   ''' Write a new dataset (db and videos/pictures). '''

#   def __init__(self, out_db_file, overwrite=False):

#     db_name = op.splitext(op.basename(out_db_file))[0]
#     if op.isabs(out_db_file):
#       logging.info('DatasetWriter: considering "%s" as absolute path.' % out_db_file)
#       out_dir = op.dirname(out_db_file)
#       self.imagedir = db_name
#       self.maskdir = self.imagedir + 'mask'
#       vimagefile = op.join(out_dir, self.imagedir + '.avi')
#       vmaskfile = op.join(out_dir, self.maskdir + '.avi')
#       logging.info('DatasetWriter: imagedir is relative to db path: "%s"' % self.imagedir)
#     else:
#       logging.info('DatasetWriter: considering "%s" as relative to CITY_PATH.' % out_db_file)
#       out_db_file = atcity(out_db_file)
#       out_dir = op.dirname(out_db_file)
#       self.imagedir = op.join(op.relpath(out_dir, os.getenv('CITY_PATH')), db_name)
#       self.maskdir = self.imagedir + 'mask'
#       vimagefile = op.abspath(op.join(os.getenv('CITY_PATH'), self.imagedir + '.avi'))
#       vmaskfile = op.abspath(op.join(os.getenv('CITY_PATH'), self.maskdir + '.avi'))
#       logging.info('DatasetWriter: imagedir is relative to CITY_PATH: "%s"' % self.imagedir)

#     if not op.exists(out_dir):
#       os.makedirs(out_dir)

#     self.video_writer = SimpleWriter(vimagefile=vimagefile, vmaskfile=vmaskfile,
#                                      unsafe=overwrite)

#     if op.exists(out_db_file):
#       if overwrite:
#         os.remove(out_db_file)
#       else:
#         raise Exception('%s already exists. A mistake?' % out_db_file)
#     self.conn = sqlite3.connect(out_db_file)
#     self.c = self.conn.cursor()
#     createDb(self.conn)

#     self.i_image = -1

#   def add_image(self, image, mask=None, timestamp=None):
#     self.i_image += 1
#     imagefile = op.join(self.imagedir, '%06d' % self.i_image)
#     height, width = image.shape[0:2]
#     if timestamp is None: timestamp = makeTimeString(datetime.now())
#     maskfile = None if mask is None else op.join(self.maskdir, '%06d' % self.i_image)
#     image_entry = (imagefile, width, height, maskfile, timestamp)

#     s = 'images(imagefile,width,height,maskfile,time)'
#     self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?)' % s, image_entry)

#     self.video_writer.imwrite(image)
#     if mask is not None: self.video_writer.maskwrite(mask)

#     return imagefile

#   def add_car(self, car_entry):
#     if len(car_entry) == 10:
#       s = 'cars(imagefile,name,x1,y1,width,height,score,yaw,pitch,color)'
#       logging.debug('Adding a new car %s' % str(car_entry))
#       self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?,?);' % s, car_entry)
#       return self.c.lastrowid
#     elif len(car_entry) == 11:
#       s = 'cars(id,imagefile,name,x1,y1,width,height,score,yaw,pitch,color)'
#       logging.debug('Adding a new car %s' % str(car_entry))
#       self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?,?,?);' % s, car_entry)
#       return
#     else:
#       raise Exception('Wrong format of car_entry.')

#   def add_match(self, car_id, match=None):
#     if match is None:
#       self.c.execute('SELECT MAX(match) FROM matches')
#       match = self.c.fetchone()[0]
#       match = match + 1 if match is not None else 0
#     s = 'matches(match,carid)'
#     logging.debug('Adding a new match %d for car_id %d' % (match, car_id))
#     self.c.execute('INSERT INTO %s VALUES (?,?);' % s, (match, car_id))
#     return match

#   def close(self):
#     self.conn.commit()
#     self.conn.close()



def cropObjectsParser(subparsers):
  parser = subparsers.add_parser('cropObjects',
    description='Crops object patches to pictures or video and saves their info as a db. '
    'All imagefiles and maskfiles in the db will be replaced with the crops, one per object.')
  parser.set_defaults(func=cropObjects)
  imgroup = parser.add_mutually_exclusive_group()
  imgroup.add_argument('--image_pictures_dir',
    help='the directory where to write mask crop pictures to.')
  imgroup.add_argument('--image_video_file',
    help='the video file where to write image crops to.')
  maskgroup = parser.add_mutually_exclusive_group()
  maskgroup.add_argument('--mask_pictures_dir',
    help='the directory where to write mask crop pictures to.')
  maskgroup.add_argument('--mask_video_file',
    help='the video file where to write mask crops to.')
  parser.add_argument('--target_width', required=True, type=int)
  parser.add_argument('--target_height', required=True, type=int)
  parser.add_argument('--edges', default='distort',
    choices={'distort', 'constant', 'background'},
    help='''"distort" distorts the patch to get to the desired ratio,
            "constant" keeps the ratio but pads the patch with zeros,
            "background" keeps the ratio but includes image background.''')
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')

def cropObjects(c, args):
  imreader = ImageryReader(rootdir=args.rootdir)

  # Create image writer.
  if args.image_video_file and args.mask_pictures_dir or args.image_pictures_dir and args.mask_video_file:
    raise ValueError('Images and masks have to be the same media -- pictures or video')
  elif args.image_video_file:
    imwriter = VideoWriter(rootdir=args.rootdir,
      vimagefile=args.image_video_file, 
      vmaskfile=args.mask_video_file,
      overwrite=args.overwrite)
  elif args.image_pictures_dir:
    imwriter = PictureWriter(rootdir=args.rootdir)
  else:
    raise ValueError('Specify either "image_video_file" or "image_pictures_dir".')

  c.execute('SELECT o.objectid,o.imagefile,o.x1,o.y1,o.width,o.height,o.name,o.score,i.maskfile,i.timestamp '
    'FROM objects AS o INNER JOIN images AS i ON i.imagefile = o.imagefile ORDER BY objectid')
  entries = c.fetchall()
  logging.debug(pformat(entries))

  c.execute('DELETE FROM images')

  for objectid,imagefile,x1,y1,width,height,name,score,maskfile,timestamp in progressbar(entries):
    logging.info ('Processing object %d from imagefile %s.' % (objectid, imagefile))
    roi = bbox2roi((x1,y1,width,height))

    # Write image.
    image = imreader.imread(imagefile)
    image = cropPatch(image, roi, args.target_height, args.target_width, args.edges)
    if args.image_video_file:
      imagefile = imwriter.imwrite(image)
    elif args.image_pictures_dir:
      imagefile = op.join(args.image_pictures_dir, '%09d.jpg' % objectid)
      imwriter.imwrite(imagefile, image)
    else:
      assert False

    # Write mask.
    if maskfile is not None:
      mask = imreader.maskread(maskfile)
      mask = cropPatch(mask, roi, args.target_height, args.target_width, args.edges)
      if args.mask_video_file:
        maskfile = imwriter.maskwrite(mask)
      elif args.mask_pictures_dir:
        maskfile = op.join(args.mask_pictures_dir, '%09d.png' % objectid)
        imwriter.imwrite(maskfile, mask)
      else:
        maskfile = None

    logging.info('Recording imagefile %s and maskfile %s.' % (imagefile, maskfile))
    c.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',
      (imagefile, args.target_width, args.target_height, maskfile, timestamp, name, score))
    
  imwriter.close()



def writeImagesParser(subparsers):
  parser = subparsers.add_parser('writeImages',
    description='Export images as a directory with pictures or as a video.')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--out_pictures_dir',
    help='the directory where to write image pictures to.')
  group.add_argument('--out_video_file',
    help='the video file where to write images to.')
  parser.add_argument('--mask_mapping_dict', 
    help='how values in maskfile are drawn. E.g. "{0: [0,0,0], 255: [128,128,30]}"')
  parser.add_argument('--mask_alpha', type=float, default=0.2,
    help='transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--with_imageid', action='store_true', help='print frame number.')
  parser.add_argument('--with_objects', action='store_true', help='draw objects on top.')
  parser.add_argument('--overwrite', action='store_true', help='overwrite video if it exists.')
  parser.set_defaults(func=writeImages)

def writeImages (c, args):
  imreader = ImageryReader(rootdir=args.rootdir)

  # For overlaying masks.
  labelmap = literal_eval(args.mask_mapping_dict) if args.mask_mapping_dict else None
  logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

  # Create writer.
  if args.out_video_file:
    imwriter = VideoWriter(rootdir=args.rootdir, vimagefile=args.out_video_file, overwrite=args.overwrite)
  elif args.out_pictures_dir:
    imwriter = PictureWriter(rootdir=args.rootdir)
  else:
    raise ValueError('Specify either "out_video_file" or "out_pictures_dir".')

  c.execute('SELECT imagefile,maskfile FROM images')
  for imagefile, maskfile in progressbar(c.fetchall()):

    logging.info ('Imagefile "%s"' % imagefile)
    image = imreader.imread(imagefile)

    # Overlay the mask.
    if maskfile is not None:
      mask = imreader.maskread(maskfile)
      image = drawMaskOnImage(image, mask, alpha=args.mask_alpha, labelmap=labelmap)
    else:
      logging.info('No mask for this image.')

    # Overlay imagefile.
    if args.with_imageid:
      drawImageId(image, imagefile)

    # Draw objects as polygons (preferred) or ROI.
    if args.with_objects:
      c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
      for object_entry, in c.fetchall():
        roi   = objectField(object_entry, 'roi')
        score = objectField(object_entry, 'score')
        name  = objectField(object_entry, 'name')
        c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid,))
        polygon_entries = c.fetchall()
        if len(polygon_entries) > 0:
          logging.info('showing object with a polygon.')
          polygon = [(polygonField(p, 'x'), polygonField(p, 'y')) for p in polygon_entries]
          drawScoredPolygon (image, polygon, label=name, score=score)
        elif roi is not None:
          logging.info('showing object with a bounding box.')
          drawScoredRoi (image, roi, label=name, score=score)
        else:
          raise Exception('Neither polygon, nor bbox is available for objectid %d' % objectid)

    if args.out_video_file:
      imwriter.imwrite(image)
    elif args.out_pictures_dir:
      imagefile = op.join(args.out_pictures_dir, op.basename(imagefile))
      imwriter.imwrite(imagefile, image)
  imwriter.close()
