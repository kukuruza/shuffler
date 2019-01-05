import os, sys, os.path as op
import numpy as np
import cv2
import logging
from ast import literal_eval
from pprint import pformat

from .util import drawTextOnImage, drawMaskOnImage, drawMaskAside
from .util import bbox2roi, drawScoredRoi, drawScoredPolygon
from .util import FONT, SCALE, FONT_SIZE, THICKNESS
from .backendDb import deleteImage, deleteObject, objectField, polygonField
from .backendMedia import MediaReader, normalizeSeparators


def add_parsers(subparsers):
  examineImagesParser(subparsers)
  examineObjectsParser(subparsers)
  examineMatchesParser(subparsers)


class KeyReader:
  ''' A mapper from keyboard buttons to actions. '''

  def __init__(self, keysmap_str):
    '''
    Args:
      key_dict:  a string that can be parsed as dict Key->Action, where: 
                  Key - a string of one char or a an int for ASCII
                  Action - a string. Actions 'exit', 'previous', and 'next' must be present.
    Returns:
      Parsed dict.
    '''
    keysmap = literal_eval(keysmap_str)
    logging.info('Keys map was parsed as: %s' % pformat(keysmap))
    for value in ['exit', 'previous', 'next']:
      if not value in keysmap.values():
        raise ValueError('"%s" must be in the values of keysmap, '
                         'now the values are: %s' % (value, keysmap.values()))
    for key in keysmap:
      if not isinstance(key, int) and not (isinstance(key, str) and len(key) == 1):
        raise ValueError('Each key in key_dict must be an integer (for ASCII code) '
                         'or a string of a single character. Got key: %s' % key)
    self.keysmap = keysmap

  def parse(self, button):
    ''' Get the corresponding action for a pressed button. '''

    if chr(button) in self.keysmap:
      # Get the character from ASCII, since character is in the keymap.
      logging.info('Found char "%s" for pressed ASCII %d in the table.' % (chr(button), button))
      button = chr(button)

    if button in self.keysmap:
      logging.info('Value for pressed "%s" is "%s".' % (str(button), self.keysmap[button]))
      return self.keysmap[button]
    else:
      logging.info('No value for pressed "%s".' % str(button))
      return None


def examineImagesParser(subparsers):
  parser = subparsers.add_parser('examineImages',
    description='Loop through images. Possibly, assign names to images.')
  parser.set_defaults(func=examineImages)
  parser.add_argument('--where_images', default='TRUE',
    help='the SQL "where" clause for the "images" table.')
  parser.add_argument('--mask_mapping_dict', 
    help='how values in maskfile are displayed. E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--mask_aside', action='store_true',
    help='Image and mask side by side.')
  group.add_argument('--mask_alpha', type=float, default=0.2,
    help='Mask will be overlaied on the image.' 
    'Transparency to overlay the label mask with, 1 means cant see the image behind the mask.')
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--with_objects', action='store_true',
    help='draw all objects on top of the image.')
  parser.add_argument('--winsize', type=int, default=500)
  parser.add_argument('--key_dict',
    default='{"-": "previous", "=": "next", " ": "next", 127: "delete", 27: "exit"}')

def examineImages (c, args):
  cv2.namedWindow("examineImages")

  c.execute('SELECT imagefile,maskfile FROM images')
  image_entries = c.fetchall()
  logging.info('%d images found.' % len(image_entries))
  if len(image_entries) == 0:
    logging.error('There are no images. Exiting.')
    return

  if args.shuffle:
    np.random.shuffle(image_entries)

  imreader = MediaReader(rootdir=args.rootdir)

  # For parsing keys.
  key_reader = KeyReader(args.key_dict)

  # For overlaying masks.
  labelmap = literal_eval(args.mask_mapping_dict) if args.mask_mapping_dict else None
  logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

  index_image = 0

  while True:  # Until a user hits the key for the "exit" action.

    imagefile, maskfile = image_entries[index_image]
    logging.info ('Imagefile "%s"' % imagefile)
    image = imreader.imread(imagefile)

    # Overlay the mask.
    if maskfile is not None:
      mask = imreader.maskread(maskfile)
      if args.mask_aside:  # Image and mask side by side
        image = drawMaskAside(image, mask, labelmap=labelmap)
      elif args.mask_alpha is not None:
        image = drawMaskOnImage(image, mask, alpha=args.mask_alpha, labelmap=labelmap)
    else:
      logging.info('No mask for this image.')

    # Put the objects on top of the image.
    if args.with_objects:
      c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile,))
      object_entries = c.fetchall()
      logging.info ('Found %d objects for image %s' % (len(object_entries), imagefile))
      for object_entry in object_entries:
        objectid     = objectField(object_entry, 'objectid')
        roi          = objectField(object_entry, 'roi')
        score        = objectField(object_entry, 'score')
        name         = objectField(object_entry, 'name')
        logging.info ('objectid: %d, roi: %s, score: %s, name: %s' % (objectid, roi, score, name))
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

    # Display an image, wait for the key from user, and parse that key.
    scale = float(args.winsize) / max(list(image.shape[0:2]))
    image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
    # Overlay imagefile.
    drawTextOnImage(image, op.basename(normalizeSeparators(imagefile)))
    # Display
    cv2.imshow('examineImages', image[:,:,::-1])
    action = key_reader.parse (cv2.waitKey(-1))
    if action is None:
      # User pressed something which does not have an action.
      continue
    elif action == 'delete':
      deleteImage (c, imagefile)
      del image_entries[index_image]
      if len(image_entries) == 0:
        logging.warning('Deleted the last image.')
        break
      index_image += 1
    elif action == 'exit':
      break
    elif action == 'previous':
      index_image -= 1
    elif action == 'next':
      index_image += 1
    else:
      # User pressed something else which has an assigned action, assume it is a new name.
      logging.info('Setting name "%s" to imagefile "%s"' % (action, imagefile))
      c.execute('UPDATE images SET name=? WHERE imagefile=?' (action, imagefile))
    index_image = index_image % len(image_entries)

  cv2.destroyWindow("examineImages")



def examineObjectsParser(subparsers):
  parser = subparsers.add_parser('examineObjects',
    description='Loop through objects.')
  parser.set_defaults(func=examineObjects)
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--where_object', default='TRUE',
    help='the SQL "where" clause for the "objects" table.')
  parser.add_argument('--winsize', type=int, default=500)
  parser.add_argument('--key_dict', 
    default='{"-": "previous", "=": "next", 27: "exit", 127: "delete", " ": "unname"}')
  # TODO: add display of mask.

def examineObjects (c, args):
  cv2.namedWindow("examineObjects")

  c.execute('SELECT COUNT(*) FROM objects WHERE (%s) ' % args.where_object)
  logging.info('Found %d objects in db.' % c.fetchone()[0])

  c.execute('SELECT DISTINCT imagefile FROM objects WHERE (%s) ' % args.where_object)
  image_entries = c.fetchall()
  logging.info('%d images found.' % len(image_entries))
  if len(image_entries) == 0:
    logging.error('There are no images. Exiting.')
    return

  if args.shuffle:
    np.random.shuffle(image_entries)

  imreader = MediaReader(rootdir=args.rootdir)

  # For parsing keys.
  key_reader = KeyReader(args.key_dict)

  index_image = 0
  index_object = 0

  # Iterating over images, because for each image we want to show all objects.
  while True:  # Until a user hits the key for the "exit" action.

    (imagefile,) = image_entries[index_image]
    logging.info ('Imagefile "%s"' % imagefile)
    image = imreader.imread(imagefile)

    c.execute('SELECT * FROM objects WHERE imagefile=? AND (%s)' % args.where_object, (imagefile,))
    object_entries = c.fetchall()
    logging.info ('Found %d objects for image %s' % (len(object_entries), imagefile))

    # Put the objects on top of the image.
    if len(object_entries) > 0:
      assert index_object < len(object_entries)
      object_entry = object_entries[index_object]
      objectid     = objectField(object_entry, 'objectid')
      roi          = objectField(object_entry, 'roi')
      score        = objectField(object_entry, 'score')
      name         = objectField(object_entry, 'name')
      logging.info ('objectid: %d, roi: %s, score: %s, name: %s' % (objectid, roi, score, name))
      c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid,))
      polygon_entries = c.fetchall()
      if len(polygon_entries) > 0:
        logging.info('showing object with a polygon.')
        polygon = [(polygonField(p, 'x'), polygonField(p, 'y')) for p in polygon_entries]
        drawScoredPolygon (image, polygon, label=None, score=score)
      elif roi is not None:
        logging.info('showing object with a bounding box.')
        drawScoredRoi (image, roi, label=None, score=score)
      else:
        raise Exception('Neither polygon, nor bbox is available for objectid %d' % objectid)
      c.execute('SELECT key,value FROM properties WHERE objectid=?', (objectid,))
      properties = c.fetchall()
      if name is not None:
        properties.append(('name', name))
      if score is not None:
        properties.append(('score', score))
      for iproperty, (key, value) in enumerate(properties):
        cv2.putText (image, '%s: %s' % (key, value), (10, SCALE * (iproperty + 1)), 
            FONT, FONT_SIZE, (0,0,0), THICKNESS)
        cv2.putText (image, '%s: %s' % (key, value), (10, SCALE * (iproperty + 1)), 
            FONT, FONT_SIZE, (255,255,255), THICKNESS-1)
        logging.info ('objectid: %d. %s = %s.' % (objectid, key, value))

    # Display an image, wait for the key from user, and parse that key.
    scale = float(args.winsize) / max(image.shape[0:2])
    image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
    cv2.imshow('examineObjects', image[:,:,::-1])
    action = key_reader.parse (cv2.waitKey(-1))
    if action is None:
      # User pressed something which does not have an action.
      continue
    elif action == 'exit':
      break
    elif action == 'previous':
      index_object -= 1
      if index_object < 0:
        index_image -= 1
        index_object = 0
    elif action == 'next':
      index_object += 1
      if index_object >= len(object_entries):
        index_image += 1
        index_object = 0
    elif action == 'delete' and len(object_entries) > 0:
      deleteObject (c, objectid)
      del object_entries[index_object]
      if index_object >= len(object_entries):
        index_image += 1
        index_object = 0
    elif action == 'unname' and len(object_entries) > 0:
      logging.info('Remove the name from objectid "%s"' % objectid)
      c.execute('UPDATE objects SET name=NULL WHERE objectid=?', (objectid,))
    elif len(object_entries) > 0:
      # User pressed something else which has an assigned action, assume it is a new name.
      objectid = objectField(object_entry, 'objectid')
      logging.info('Setting name "%s" to objectid "%s"' % (action, objectid))
      c.execute('UPDATE objects SET name=? WHERE objectid=?', (action, objectid))
    index_image = index_image % len(image_entries)

  cv2.destroyWindow("examineObjects")


def examineMatchesParser(subparsers):
  parser = subparsers.add_parser('examineMatches',
    description='''Browse through database and see car bboxes on top of images.
                   Any key will scroll to the next image.''')
  parser.set_defaults(func=examineMatches)
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--winsize', type=int, default=500)
  parser.add_argument('--key_dict', 
      default='{"-": "previous", "=": "next", " ": "next", 27: "exit"}')

def examineMatches (c, args):
  cv2.namedWindow("examineMatches")

  c.execute('SELECT DISTINCT(match) FROM matches')
  match_entries = c.fetchall()

  if args.shuffle:
    np.random.shuffle(match_entries)

  imreader = MediaReader(rootdir=args.rootdir)

  # For parsing keys.
  key_reader = KeyReader(args.key_dict)

  index_match = 0

  # Iterating over images, because for each image we want to show all objects.
  while True:  # Until a user hits the key for the "exit" action.

    (match,) = match_entries[index_match]
    c.execute('SELECT * FROM objects WHERE objectid IN '
              '(SELECT objectid FROM matches WHERE match=?)', (match,))
    object_entries = c.fetchall()
    logging.info ('Found %d objects for match %d' % (len(object_entries), match))

    images = []
    for object_entry in object_entries:
      imagefile = objectField(object_entry, 'imagefile')
      objectid  = objectField(object_entry, 'objectid')
      roi       = objectField(object_entry, 'roi')
      score     = objectField(object_entry, 'score')

      image = imreader.imread(imagefile)
      drawScoredRoi(image, roi, score=score)

      scale = float(args.winsize) / max(image.shape[0:2])
      image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
      images.append(image)

    # Assume all images have the same size for now.
    # Assume there are not so many matches.
    image = np.hstack(images)

    # Display an image, wait for the key from user, and parse that key.
    cv2.imshow('examineMatches', image[:,:,::-1])
    action = key_reader.parse (cv2.waitKey(-1))
    if action is None:
      # User pressed something which does not have an action.
      continue
    elif action == 'exit':
      break
    elif action == 'previous':
      index_match -= 1
    elif action == 'next':
      index_match += 1
    else:
      # User pressed something else which has an assigned action, assume it is a new name.
      logging.info('Setting name "%s" to imagefile "%s"' % (action, imagefile))
      c.execute('UPDATE images SET name=? WHERE imagefile=?' (action, imagefile))
    index_match = index_match % len(match_entries)

  cv2.destroyWindow("examineMatches")
