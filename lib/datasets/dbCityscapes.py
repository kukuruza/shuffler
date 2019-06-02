import os, sys, os.path as op
import numpy as np
import cv2
import collections
import logging
from glob import glob
import shutil
import simplejson as json
import sqlite3
from progressbar import progressbar
from pprint import pformat

from ..backendDb import objectField, deleteObject
from ..backendMedia import MediaReader, getPictureSize
from ..util import drawScoredRoi, drawScoredPolygon, drawMaskAside


def add_parsers(subparsers):
  importCityscapesParser(subparsers)


def _importJson(c, jsonpath, imagefile, imheight, imwidth):
  with open(jsonpath) as f:
    image_dict = json.load(f)
  if image_dict['imgHeight'] != imheight or image_dict['imgWidth'] != imwidth:
    raise Exception('Image size is inconsistent to json: %dx%d vs %dx%d' %
      (image_dict['imgHeight'], image_dict['imgWidth'], imheight, imwidth))
  objects = image_dict['objects']
  for object_ in objects:
    name = object_['label']
    polygon = object_['polygon']
    x1 = int(np.min([point[0] for point in polygon]))
    y1 = int(np.min([point[1] for point in polygon]))
    x2 = int(np.max([point[0] for point in polygon]))
    y2 = int(np.max([point[1] for point in polygon]))
    width = x2 - x1
    height = y2 - y1
    c.execute('INSERT INTO objects(imagefile,x1,y1,width,height,name) VALUES (?,?,?,?,?,?)',
                                  (imagefile,x1,y1,width,height,name))
    objectid = c.lastrowid
    for point in polygon:
      s = 'INSERT INTO polygons(objectid,x,y) VALUES (?,?,?)'
      c.execute(s, (objectid,point[0],point[1]))



def importCityscapesParser(subparsers):
  parser = subparsers.add_parser('importCityscapes',
    description='Import Cityscapes annotations into the database. '
    'Images are assumed to be from leftImg8bit.')
  parser.set_defaults(func=importCityscapes)
  parser.add_argument('--cityscapes_dir', required=True,
      help='Root directory of Cityscapes. '
      'It should contain subdirs "gtFine_trainvaltest", "leftImg8bit_trainvaltest", etc.')
  parser.add_argument('--splits', nargs='+', default=['train', 'test', 'val'],
      choices=['train', 'test', 'val', 'train_extra', 'demoVideo'],
      help='Splits to be parsed.')
  parser.add_argument('--type', default='gtFine', choices=['gtFine', 'gtCoarse'],
      help='Which annotations to parse. '
      'Will not parse both to keep the logic straightforward.')
  parser.add_argument('--mask_type', choices=['labelIds', 'instanceIds', 'color'],
      help='Which mask to import, if any.')
  parser.add_argument('--with_display', action='store_true')

def importCityscapes (c, args):
  if args.with_display:
    imreader = MediaReader(args.rootdir)

  logging.info('Will load splits: %s' % args.splits)
  logging.info('Will load json type: %s' % args.type)
  logging.info('Will load mask type: %s' % args.mask_type)

  if not op.exists(args.cityscapes_dir):
    raise Exception('Cityscape directory "%s" does not exist' % args.cityscapes_dir)

  # Directory accessed by label type and by split. 
  dirs_by_typesplit = {}

  for type_ in [args.type, 'leftImg8bit']:
    type_dir_template = op.join(args.cityscapes_dir, '%s*' % type_)
    for type_dir in [x for x in glob(type_dir_template) if op.isdir(x)]:
      logging.debug('Looking for splits in %s' % type_dir)
      for split in args.splits:
        typesplit_dir = op.join(type_dir, type_, split)
        if op.exists(typesplit_dir):
          logging.debug('Found split %s in %s' % (split, type_dir))
          # Add the info into the main dictionary "dirs_by_typesplit".
          if split not in dirs_by_typesplit:
            dirs_by_typesplit[split] = {}
          dirs_by_typesplit[split][type_] = typesplit_dir
  logging.info('Found the following directories: \n%s' % pformat(dirs_by_typesplit))

  for split in args.splits:
    # List of cities.
    assert 'leftImg8bit' in dirs_by_typesplit[split]
    leftImg8bit_dir = dirs_by_typesplit[split]['leftImg8bit']
    cities = os.listdir(leftImg8bit_dir)
    cities = [x for x in cities if op.isdir(op.join(leftImg8bit_dir, x))]
    logging.info('Found %d cities in %s' % (len(cities), leftImg8bit_dir))

    for city in cities:
      image_names = os.listdir(op.join(leftImg8bit_dir, city))
      logging.info('In split "%s", city "%s" has %d images' % (split, city, len(image_names)))

      for image_name in image_names:

        # Get the image path.
        image_path = op.join(leftImg8bit_dir, city, image_name)
        name_parts = op.splitext(image_name)[0].split('_')
        if len(name_parts) != 4:
          raise Exception('Expect to have 4 parts in the name of image "%s"' % image_name)
        if name_parts[0] != city:
          raise Exception('The first part of name of image "%s" '
            'is expected to be city "%s".' % (image_name, city))
        if name_parts[3] != 'leftImg8bit':
          raise Exception('The last part of name of image "%s" '
            'is expected to be city "leftImg8bit".' % image_name)
        imheight, imwidth = getPictureSize(image_path)
        imagefile = op.relpath(image_path, args.rootdir)
        c.execute('INSERT INTO images(imagefile,width,height) VALUES (?,?,?)',
             (imagefile,imwidth,imheight))

        # Get the json label.
        if args.type in dirs_by_typesplit[split]:
          city_dir = op.join(dirs_by_typesplit[split][args.type], city)
          if op.exists(city_dir):

            json_name = '_'.join(name_parts[:3] + [args.type, 'polygons']) + '.json'
            json_path = op.join(city_dir, json_name)
            if op.exists(json_path):
              _importJson(c, json_path, imagefile, imheight, imwidth)

            if args.mask_type is not None:
              mask_name = '_'.join(name_parts[:3] + [args.type, args.mask_type]) + '.png'
              mask_path = op.join(city_dir, mask_name)
              if op.exists(mask_path):
                maskfile = op.relpath(mask_path, args.rootdir)
                c.execute('UPDATE images SET maskfile=? WHERE imagefile=?', (maskfile,imagefile))

        if args.with_display:
          img = imreader.imread(imagefile)
          c.execute('SELECT objectid,name FROM objects WHERE imagefile=?', (imagefile,))
          for objectid,name in c.fetchall():
            # Draw polygon.
            c.execute('SELECT x,y FROM polygons WHERE objectid=?', (objectid,))
            polygon = c.fetchall()
            drawScoredPolygon (img, [(int(pt[0]),int(pt[1])) for pt in polygon], name)
          cv2.imshow('importCityscapes', img[:,:,::-1])
          if cv2.waitKey(-1) == 27:
            args.with_display = False
            cv2.destroyWindow('importCityscapes')

  # Statistics.
  c.execute('SELECT COUNT(1) FROM images')
  logging.info('Imported %d images' % c.fetchone()[0])
  c.execute('SELECT COUNT(1) FROM images WHERE maskfile IS NOT NULL')
  logging.info('Imported %d masks' % c.fetchone()[0])
  c.execute('SELECT COUNT(DISTINCT(imagefile)) FROM objects')
  logging.info('Objects are found in %d images' % c.fetchone()[0])
