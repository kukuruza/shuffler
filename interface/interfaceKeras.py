import sys, os.path as op
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
import numpy as np
import argparse
import logging
import pprint
import keras.utils

from interface import utils
from lib.backend import backendDb
from lib.backend import backendMedia


class BareImageGenerator(keras.utils.Sequence):
    ''' Generates only images for Keras. '''
    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_image='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 batch_size=1,
                 shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not op.exists(db_file):
            raise ValueError('db_file does not exist: %s' % db_file)

        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' %
                       where_image)
        self.image_entries = self.c.fetchall()

        self.imreader = backendMedia.MediaReader(rootdir=rootdir)

        self.on_epoch_end()

    def __len__(self):
        ''' Denotes the number of batches per epoch. '''
        return int(np.ceil(len(self.image_entries) / self.batch_size))

    def close(self):
        self.conn.close()

    def _load_image(self, image_entry):
        logging.debug('Reading image "%s"' %
                      backendDb.imageField(image_entry, 'imagefile'))
        img = self.imreader.imread(
            backendDb.imageField(image_entry, 'imagefile'))
        return img

    def _loadEntry(self, index):
        image_entry = self.image_entries[index]
        img = self._load_image(image_entry)
        imagefile = backendDb.imageField(image_entry, 'imagefile')
        return {'image': img, 'imagefile': imagefile}

    def __getitem__(self, index):
        ''' Generate one batch of data. '''

        # Generate indexes of the batch.
        if (index + 1) * self.batch_size < len(self.image_entries):
            last_index = (index + 1) * self.batch_size
        else:
            last_index = len(self.image_entries)
        indexes = self.indexes[index * self.batch_size:last_index]
        Xs = [self._loadEntry(index) for index in indexes]

        # A list of dicts to a dict of lists, for both Xs and Ys.
        Xs = {k: [dic[k] for dic in Xs] for k in Xs[0]}

        return Xs

    def on_epoch_end(self):
        ''' Updates indexes after each epoch. '''
        self.indexes = np.arange(len(self.image_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class ImageGenerator(keras.utils.Sequence):
    ''' Generates images with all the objects for Keras. '''
    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_image='TRUE',
                 where_object='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 batch_size=32,
                 shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not op.exists(db_file):
            raise ValueError('db_file does not exist: %s' % db_file)

        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' %
                       where_image)
        self.image_entries = self.c.fetchall()

        self.imreader = backendMedia.MediaReader(rootdir=rootdir)
        self.where_object = where_object

        self.on_epoch_end()

    def __len__(self):
        ''' Denotes the number of batches per epoch. '''
        return int(np.floor(len(self.image_entries) / self.batch_size))

    def close(self):
        self.conn.close()

    def _load_image(self, image_entry):
        logging.debug('Reading image "%s"' %
                      backendDb.imageField(image_entry, 'imagefile'))
        img = self.imreader.imread(
            backendDb.imageField(image_entry, 'imagefile'))
        if backendDb.imageField(image_entry, 'maskfile') is None:
            mask = None
        else:
            mask = self.imreader.maskread(
                backendDb.imageField(image_entry, 'maskfile'))
        return img, mask

    def _loadEntry(self, index):
        image_entry = self.image_entries[index]
        img, mask = self._load_image(image_entry)

        imagefile = backendDb.imageField(image_entry, 'imagefile')
        imagename = backendDb.imageField(image_entry, 'name')

        self.c.execute(
            'SELECT x1,y1,width,height,name FROM objects '
            'WHERE imagefile=? AND (%s)' % self.where_object, (imagefile, ))
        object_entries = self.c.fetchall()

        return {'input': img}, \
               {'mask': mask, 'objects': object_entries,
                'imagefile': imagefile, 'name': imagename}

    def __getitem__(self, index):
        ''' Generate one batch of data. '''

        # Generate indexes of the batch.
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        Xs, Ys = [self._loadEntry(index) for index in indexes]

        # A list of dicts to a dict of lists, for both Xs and Ys.
        Xs = {k: [dic[k] for dic in Xs] for k in Xs[0]}
        Ys = {k: [dic[k] for dic in Ys] for k in Ys[0]}

        return Xs, Ys

    def on_epoch_end(self):
        ''' Updates indexes after each epoch. '''

        self.indexes = np.arange(len(self.image_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class ObjectGenerator(keras.utils.Sequence):
    ''' Items of a dataset are objects. '''
    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_object='TRUE',
                 copy_to_memory=True,
                 batch_size=32,
                 shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle

        if not op.exists(db_file):
            raise ValueError('db_file does not exist: %s' % db_file)

        self.conn = utils.openConnection(db_file,
                                         copy_to_memory=copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' %
                       where_object)
        self.object_entries = self.c.fetchall()

        self.imreader = backendMedia.MediaReader(rootdir=rootdir)

        self.on_epoch_end()

    def close(self):
        self.conn.close()

    def __len__(self):
        return len(self.object_entries)

    def _loadEntry(self, index):
        object_entry = self.object_entries[index]

        objectid = backendDb.objectField(object_entry, 'objectid')
        imagefile = backendDb.objectField(object_entry, 'imagefile')
        name = backendDb.objectField(object_entry, 'name')

        self.c.execute('SELECT maskfile FROM images WHERE imagefile=?',
                       (imagefile, ))
        maskfile = self.c.fetchone()[0]

        logging.debug('Reading object %d from %s imagefile' %
                      (objectid, imagefile))

        img = self.imreader.imread(imagefile)
        mask = self.imreader.maskread(
            maskfile) if maskfile is not None else None

        roi = backendDb.objectField(object_entry, 'roi')
        logging.debug('Roi: %s' % roi)
        img = img[roi[0]:roi[2], roi[1]:roi[3]]
        mask = mask[roi[0]:roi[2], roi[1]:roi[3]] if mask is not None else None

        X = {'image': img}
        Y = {
            'mask': mask,
            'class': name,
            'imagefile': imagefile,
            'objectid': objectid
        }

        # Add properties.
        self.c.execute('SELECT key,value FROM properties WHERE objectid=?',
                       (objectid, ))
        for key, value in self.c.fetchall():
            X[key] = value

        return X, Y

    def __getitem__(self, index):
        ''' Generate one batch of data. '''

        # Generate indexes of the batch.
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        Xs, Ys = [self._loadEntry(index) for index in indexes]

        # A list of dicts to a dict of lists, for both Xs and Ys.
        Xs = {k: [dic[k] for dic in Xs] for k in Xs[0]}
        Ys = {k: [dic[k] for dic in Ys] for k in Ys[0]}

        return Xs, Ys

    def on_epoch_end(self):
        ''' Updates indexes after each epoch. '''
        self.indexes = np.arange(len(self.object_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_db_file', required=True)
    parser.add_argument('--rootdir', required=True)
    parser.add_argument('--dataset_type',
                        required=True,
                        choices=['bare', 'image', 'object'])
    args = parser.parse_args()

    if args.dataset_type == 'bare':
        dataset = BareImageGenerator(args.in_db_file, rootdir=args.rootdir)
        item = dataset.__getitem__(1)
        pprint.pprint(item)

    elif args.dataset_type == 'image':
        dataset = ImageGenerator(args.in_db_file, rootdir=args.rootdir)
        item = dataset.__getitem__(1)
        pprint.pprint(item)

    elif args.dataset_type == 'object':
        dataset = ObjectGenerator(args.in_db_file, rootdir=args.rootdir)
        item = dataset.__getitem__(1)
        pprint.pprint(item)
