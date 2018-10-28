import os, sys, os.path as op
import numpy as np
import logging
from progressbar import progressbar
from pprint import pformat
import sqlite3
import argparse
from datetime import datetime

from backendDb import createDb, makeTimeString
from backendImages import VideoWriter, PictureWriter


class DatasetVideoWriter:
  ''' Write a new dataset (db and videos). '''
  # TODO:add writing to pictures.

  def __init__(self, out_db_file, rootdir='.', overwrite=False):

    db_name = op.splitext(op.basename(out_db_file))[0]
    outdir = op.dirname(out_db_file)
    self.imagedir = op.join(op.relpath(outdir, rootdir), db_name)
    self.maskdir = self.imagedir + 'mask'
    vimagefile = op.abspath(op.join(rootdir, self.imagedir + '.avi'))
    vmaskfile = op.abspath(op.join(rootdir, self.maskdir + '.avi'))

    if not op.exists(outdir):
      os.makedirs(outdir)

    self.imwriter = VideoWriter(vimagefile=vimagefile, vmaskfile=vmaskfile,
        overwrite=overwrite)

    if op.exists(out_db_file):
      if overwrite:
        os.remove(out_db_file)
      else:
        raise ValueError('"%s" already exists.' % out_db_file)
    self.conn = sqlite3.connect(out_db_file)
    self.c = self.conn.cursor()
    createDb(self.conn)

    self.i_image = -1

  def addImage(self, image=None, mask=None, imagefile=None, timestamp=None, name=None, width=None, height=None):
    self.i_image += 1
    if image is None and imagefile is None or image is not None and imagefile is not None:
      raise ValueError('Exactly one of "image" or "imagefile" must be non-None')
    if imagefile is None:
      imagefile = op.join('%s.avi' % self.imagedir, '%06d' % self.i_image)
      self.imwriter.imwrite(image)
    if mask is not None:
      self.imwriter.maskwrite(mask)

    if image is not None:
        height, width = image.shape[0:2]
    if timestamp is None:
      timestamp = makeTimeString(datetime.now())
    maskfile = None if mask is None else op.join('%s.avi' % self.maskdir, '%06d' % self.i_image)
    image_entry = (imagefile, width, height, maskfile, timestamp, name)

    s = 'images(imagefile,width,height,maskfile,timestamp,name)'
    logging.debug('Writing %s to %s' % (image_entry, s))
    self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)' % s, image_entry)

    return imagefile

  def addObject(self, object_dict):
    imagefile = object_dict['imagefile']
    x1 = object_dict['x1']
    y1 = object_dict['y1']
    width = object_dict['width']
    height = object_dict['height']
    name = object_dict['name']
    score = object_dict['score']
    if 'objectid' in object_dict:
      objectid = object_dict['objectid']
      self.c.execute('INSERT INTO objects(objectid,imagefile,x1,y1,width,height,name,score) '
             'VALUES (?,?,?,?,?,?,?,?)', (objectid,imagefile,x1,y1,width,height,name,score))
      return self.c.lastrowid
    else:
      self.c.execute('INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
             'VALUES (?,?,?,?,?,?,?,?)', (imagefile,x1,y1,width,height,name,score))
      objectid = self.c.lastrowid
    properties = []
    for key in object_dict:
      if key not in ['objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'name', 'score']:
        self.c.execute('INSERT INTO properties(objerctid,key,value) VALUES (?,?,?)',
          (objectid, key, object_dict[key]))
    

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
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()

  writer = DatasetVideoWriter(args.out_db_file, rootdir=args.rootdir,
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

