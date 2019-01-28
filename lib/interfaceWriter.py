import os, sys, os.path as op
import numpy as np
import logging
from progressbar import progressbar
from pprint import pformat
import sqlite3
import argparse
from datetime import datetime

from backendDb import createDb, makeTimeString
from backendMedia import MediaWriter


class DatasetWriter:
  ''' Write a new dataset (db and videos). '''
  # TODO:add writing to pictures.

  def __init__(self, out_db_file, rootdir='.', media='video',
               image_path=None, mask_path=None, overwrite=False):

    self.rootdir = rootdir

    db_file_no_ext = op.splitext(out_db_file)[0]
    if image_path is None:
      if media == 'video':
        image_path = '%s.avi' % db_file_no_ext
      else:
        image_path = db_file_no_ext
    if mask_path is None:
      if media == 'video':
        mask_path = '%smask.avi' % db_file_no_ext
      else:
        mask_path = '%smask' % db_file_no_ext

    outdir = op.dirname(out_db_file)
    if not op.exists(outdir):
      os.makedirs(outdir)

    self.imwriter = MediaWriter(
      media_type=media,
      rootdir=rootdir,
      image_media=op.abspath(op.join(rootdir, image_path)),
      mask_media=op.abspath(op.join(rootdir, mask_path)),
      overwrite=overwrite)

    if op.exists(out_db_file):
      if overwrite:
        os.remove(out_db_file)
      else:
        raise ValueError('"%s" already exists.' % out_db_file)
    self.conn = sqlite3.connect(out_db_file)
    self.c = self.conn.cursor()
    createDb(self.conn)

  def addImage(self, image=None, mask=None, imagefile=None, timestamp=None, name=None, width=None, height=None):
    ''' Imagefile should be provided when "image" is not provided or written. '''
    if image is None == imagefile is None:
      raise ValueError('Exactly one of "image" or "imagefile" must be None')
    # If image is provided.
    if image is not None:
      imagefile = self.imwriter.imwrite(image)
      height, width = image.shape[0:2]
    maskfile = self.imwriter.maskwrite(mask) if mask is not None else None

    if timestamp is None:
      timestamp = makeTimeString(datetime.now())
    image_entry = (imagefile, width, height, maskfile, timestamp, name)

    s = 'images(imagefile,width,height,maskfile,timestamp,name)'
    logging.debug('Writing %s to %s' % (image_entry, s))
    self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)' % s, image_entry)
    return imagefile

  def addObject(self, object_dict):
    imagefile = object_dict['imagefile']
    x1 = object_dict['x1'] if 'x1' in object_dict else None
    y1 = object_dict['y1'] if 'y1' in object_dict else None
    width = object_dict['width'] if 'width' in object_dict else None
    height = object_dict['height'] if 'height' in object_dict else None
    name = object_dict['name'] if 'name' in object_dict else None
    score = object_dict['score'] if 'score' in object_dict else None
    if 'objectid' in object_dict:
      objectid = object_dict['objectid']
      self.c.execute('INSERT INTO objects(objectid,imagefile,x1,y1,width,height,name,score) '
             'VALUES (?,?,?,?,?,?,?,?)', (objectid,imagefile,x1,y1,width,height,name,score))
    else:
      self.c.execute('INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
               'VALUES (?,?,?,?,?,?,?)', (imagefile,x1,y1,width,height,name,score))
      objectid = self.c.lastrowid
    for key in object_dict:
      if key not in ['objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'name', 'score']:
        self.c.execute('INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
          (objectid, key, str(object_dict[key])))
    return objectid
    

  def addMatch(self, objectid, match=None):
    if match is None:
      self.c.execute('SELECT MAX(match) FROM matches')
      match = self.c.fetchone()[0]
      match = match + 1 if match is not None else 0
    s = 'matches(match,carid)'
    logging.debug('Adding a new match %d for objectid %d' % (match, objectid))
    self.c.execute('INSERT INTO %s VALUES (?,?);' % s, (match, objectid))
    return match

  def close(self):
    self.conn.commit()
    self.c.execute('SELECT COUNT(1) FROM images')
    logging.info('Wrote the total of %d entries to images.' % self.c.fetchone()[0])
    self.conn.close()



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--out_db_file', required=True)
  parser.add_argument('--rootdir', required=True)
  parser.add_argument('--image_path')
  parser.add_argument('--mask_path')
  parser.add_argument('--media', choices=['video', 'pictures'])
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  writer = DatasetWriter(args.out_db_file, rootdir=args.rootdir,
      media=args.media,
      image_path=args.image_path, mask_path=args.mask_path,
      overwrite=args.overwrite)
  writer.addImage(np.zeros((100,100,3), dtype=np.uint8),
             mask=np.ones((100,100), dtype=np.uint8))
  writer.close()

  conn = sqlite3.connect(args.out_db_file)
  cursor = conn.cursor()
  cursor.execute('SELECT * FROM images')
  image_entries = cursor.fetchall()
  conn.close()
  print (image_entries)

