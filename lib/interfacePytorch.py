import sys, os, os.path as op
import numpy as np
import argparse
import logging
import sqlite3
from pprint import pformat
from torch.utils.data import Dataset

from backendDb import imageField, objectField
from backendMedia import MediaReader


class ImagesDataset(Dataset):
  ''' Items of a dataset are images. '''

  def __init__(self, db_file, rootdir='.', where_image='TRUE', where_object='TRUE'):
    from torch.utils.data import Dataset

    try:
      self.conn = sqlite3.connect('file:%s?mode=ro' % db_file, uri=True)
    except TypeError:
      logging.debug('This Python version does not support connecting to SQLite by uri, '
      'will connect in regular mode (without readonly.)')
      self.conn = sqlite3.connect(db_file)
    self.c = self.conn.cursor()
    self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' % where_image)
    self.image_entries = self.c.fetchall()

    self.imreader = MediaReader(rootdir=rootdir)
    self.where_object = where_object

  def close(self):
    self.conn.close()

  def _load_image(self, image_entry):
    logging.debug ('Reading image "%s"' % imageField(image_entry, 'imagefile'))
    img = self.imreader.imread (imageField(image_entry, 'imagefile'))
    if imageField(image_entry, 'maskfile') is None:
      mask = None
    else:
      mask = self.imreader.maskread (imageField(image_entry, 'maskfile'))
    return img, mask

  def __len__(self):
    return len(self.image_entries)

  def __getitem__(self, index):
    '''Used to train for the detection/segmentation task.
    Args:
      index:      an index of an image in the dataset
    Returns a dict with keys:
      image:      np.uint8 array corresponding to an image
      mask:       np.uint8 array corresponding to a mask if exists, or None
      objects:    np.int32 array of shape [Nx5]. Each row is x1,y1,width,height,classname
      imagefile:  the image id
    '''
    image_entry = self.image_entries[index]
    img, mask = self._load_image(image_entry)

    imagefile = imageField(image_entry, 'imagefile')

    self.c.execute('SELECT x1,y1,width,height,name FROM objects WHERE imagefile=? AND (%s)' %
      self.where_object, (imagefile,))
    object_entries = self.c.fetchall()

    return {'image': img, 'mask': mask, 'objects': object_entries, 'imagefile': imagefile}



class ObjectsDataset:
  ''' Items of a dataset are objects. '''

  def __init__(self, db_file, rootdir='.', where_object='TRUE'):

    try:
      self.conn = sqlite3.connect('file:%s?mode=ro' % db_file, uri=True)
    except TypeError:
      logging.info('This Python version does not support connecting to SQLite by uri.')
      self.conn = sqlite3.connect(db_file)
    self.c = self.conn.cursor()
    self.c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' % where_object)
    self.object_entries = self.c.fetchall()

    self.imreader = MediaReader(rootdir=rootdir)

  def close(self):
    self.conn.close()

  def __len__(self):
    return len(self.object_entries)

  def __getitem__(self, index):
    '''Used to train for classification / segmentation of individual objects.
    Args:
      index:      an index of an image in the dataset
    Returns a dict with keys:
      image:      np.uint8 array corresponding to an image
      mask:       np.uint8 array corresponding to a mask if exists, or None
      class:      a string with object class name
      imagefile:  the image id
      all key-value pairs from the "properties" table
    '''
    object_entry = self.object_entries[index]

    objectid = objectField(object_entry, 'objectid')
    imagefile = objectField(object_entry, 'imagefile')
    name = objectField(object_entry, 'name')

    self.c.execute('SELECT maskfile FROM images WHERE imagefile=?', (imagefile,))
    maskfile = self.c.fetchone()[0]

    logging.debug('Reading object %d from %s imagefile' % (objectid, imagefile))

    img = self.imreader.imread(imagefile)
    mask = self.imreader.maskread(maskfile) if maskfile is not None else None

    roi = objectField(object_entry, 'roi')
    logging.debug('Roi: %s' % roi)
    img = img[roi[0]:roi[2], roi[1]:roi[3]]
    mask = mask[roi[0]:roi[2], roi[1]:roi[3]] if mask is not None else None

    item = {'image': img, 'mask': mask, 'class': name, 'imagefile': imagefile}

    # Add properties.
    self.c.execute('SELECT key,value FROM properties WHERE objectid=?', (objectid,))
    for key, value in self.c.fetchall():
      item[key] = value

    return item
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--in_db_file', required=True)
  parser.add_argument('--rootdir', required=True)
  parser.add_argument('--dataset_type', required=True, choices=['images', 'objects'])
  args = parser.parse_args()

  if args.dataset_type == 'images':
    dataset = ImagesDataset(args.in_db_file, rootdir=args.rootdir)
    item = dataset.__getitem__(1)
    print (pformat(item))

  elif args.dataset_type == 'objects':
    dataset = ObjectsDataset(args.in_db_file, rootdir=args.rootdir)
    item = dataset.__getitem__(1)
    print (pformat(item))
