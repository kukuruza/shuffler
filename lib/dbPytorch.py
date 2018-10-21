''' Pytorch dataset binding to .db data. '''

import sys, os, os.path as op
import numpy as np
import argparse
import logging
from torch.utils.data import Dataset
from backendDb import imageField, objectField
from backendImages import ImageryReader


class CityimagesDataset(Dataset):
  def __init__(self, db_file, where_image='TRUE', where_object='TRUE'):
    from torch.utils.data import Dataset

    self.conn = sqlite.connect(db_file)
    self.c = self.conn.cursor()
    self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' % image_constraint)
    self.image_entries = self.c.fetchall()

    self.imreader = ImageryReader()
    self.where_object = where_object


  def close(self):
    self.conn.close()

  def _load_image(self, image_entry):
    logging.debug ('CityImagesDataset: reading %s' % 
            imageField(image_entry, 'imagefile'))
    im = self.image_reader.imread (imageField(image_entry, 'imagefile'))
    if imageField(image_entry, 'maskfile') is None:
      mask = None
    else:
      mask = self.image_reader.maskread (imageField(image_entry, 'maskfile'))
    assert im is not None, image_entry
    assert len(im.shape) == 3 and im.shape[2] == 3, 'Change code for non 3-channel.'
    im = im[:,:,::-1]   # BGR to RGB.
    return im, mask

  def __len__(self):
    return len(self.image_entries)

  def __getitem__(self, index):
    '''Used to train detection/segmentation.
    Returns:
      im:     np.uint8 array of shape [imheight, imwidth, 3]
      gt:     np.uint8 array of shape [imheight, imwidth, 1]
              and values in range [0,255], where
              [imheight, imwidth] are image and mask ogirinal dimensions.
      boxes:  np.int32 array of shape [4]
    '''
    image_entry = self.image_entries[index]
    im, mask = self._load_image(image_entry)

    imagefile = imageField(image_entry, 'imagefile')

    s = 'SELECT x1,y1,width,height FROM cars WHERE imagefile="%s" AND (%s)' % (imagefile, self.car_constraint)
    logging.debug('dbDataset request: %s' % s)
    self.c.execute(s)
    bboxes = self.c.fetchall()

    item = {'image': im, 'mask': mask, 'bboxes': bboxes, 'imagefile': imagefile}

    if self.use_maps:
        item['sizemap'] = self.sizemap_cache[imagefile]
        item['pitchmap'] = self.pitchmap_cache[imagefile]
        #azimuthmap = self.azimuthmap_cache[imagefile]
        # FIXME: What to do with azimuth - bin it?

    return item


class CitycarsDataset:
  '''
  One item is one car, rather than image with multiple cars.
  Car can be cropped.
  '''
  def __init__(self, db_file, fraction=1., image_constraint='1', car_constraint='1',
               crop_car=True, with_mask=False):

    self.conn = loadToMemory(db_file)
    self.c = self.conn.cursor()
    s = ('SELECT * FROM cars WHERE %s AND ' % car_constraint +
         'imagefile IN (SELECT imagefile FROM images WHERE %s) ' % image_constraint +
         'ORDER BY imagefile')
    self.c.execute(s)
    self.car_entries = self.c.fetchall()

    self.fraction = fraction
    self.crop_car = crop_car
    self.image_reader = ReaderVideo()
    self.with_mask = with_mask

  def close(self):
    self.conn.close()

  def _load_image(self, car_entry):
    logging.debug ('CitycarsDataset: reading car %d from %s imagefile' % 
            (carField(car_entry, 'id'), carField(car_entry, 'imagefile')))
    im = self.image_reader.imread (carField(car_entry, 'imagefile'))
    if self.with_mask:
      imagefile = carField(car_entry, 'imagefile')
      self.c.execute('SELECT maskfile FROM images WHERE imagefile=?', (imagefile,))
      maskfile = self.c.fetchone()[0]
      if maskfile is None:
        mask = None
      else:
        mask = self.image_reader.maskread(maskfile)
    else:
      mask = None
    assert len(im.shape) == 3 and im.shape[2] == 3, 'Change code for non 3-channel.'
    im = im[:,:,::-1]   # BGR to RGB.
    if self.crop_car:
      roi = carField(car_entry, 'roi')
      im = im[roi[0]:roi[2], roi[1]:roi[3]]
      mask = mask[roi[0]:roi[2], roi[1]:roi[3]] if mask is not None else None
    return {'image': im, 'mask': mask, 'entry': car_entry}

  def __len__(self):
    return int(len(self.car_entries) * self.fraction)

  def __getitem__(self, index):
    '''
    Returns dict with fields:
      im: np.uint8 array of shape [imheight, imwidth, 3]
      gt: car_entry
    '''
    car_entry = self.car_entries[index]
    return self._load_image(car_entry)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--in_db_file', required=True)
  parser.add_argument('--dataset_type', required=True,
      choices=['images', 'cars'])
  args = parser.parse_args()

  if args.dataset_type == 'images':
    dataset = CityimagesDataset(args.in_db_file, fraction=0.01)
    im, _ = dataset.__getitem__(1)
    print im.shape

  elif args.dataset_type == 'cars':
    dataset = CitycarsDataset(args.in_db_file, fraction=0.005)
    im, car_entry = dataset.__getitem__(1)
    print im.shape
