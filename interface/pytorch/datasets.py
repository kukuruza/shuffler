import sys, os.path as op
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
import argparse
import logging
import pprint
import numpy as np
import torch.utils.data

from interface import utils
from lib.backend import backendDb
from lib.backend import backendMedia


class ImageDataset(torch.utils.data.Dataset):
    '''
    Items of a dataset are images.
    '''
    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_image='TRUE',
                 where_object='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 used_keys=None,
                 transform_group=None):
        '''
        Args:
            db_file:        (string) A path to an sqlite3 database file.
            rootdir:        (string) A root path, is pre-appended to "imagefile"
                            entries of "images" table in the sqlite3 database.
            where_image:    (string) The WHERE part of the SQL query on the 
                            "images" table, as in: 
                            "SELECT * FROM images WHERE ${where_image};"
                            Allows to query only needed images.
            where_object:   (string) The WHERE part of the SQL query on the 
                            "objects" table, as in: 
                            "SELECT * FROM objects WHERE ${where_object};"
                            Allows to query only needed objects for each image.
            mode:           ("r" or "w") The readonly or write-read mode to open 
                            the database. The default is "r", use "w" only if 
                            you plan to call addRecord().
            copy_to_memory: (bool) Copies database into memory. 
                            Only for mode="r". Should be used for python2. 
            transform_group: (a callable or a dict string -> callable) 
                            Transform(s) to be applied on a sample.
                            If it is a callable, it is applied to the sample.
                            If it is a dict, each key is matched to a key in the
                            sample, and callables are called on the respective
                            elements.
        '''

        self.mode = mode
        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' %
                       where_image)
        self.image_entries = self.c.fetchall()

        self.imreader = backendMedia.MediaReader(rootdir=rootdir)
        self.where_object = where_object

        self.used_keys = used_keys
        self.transform_group = transform_group

    def close(self):
        if self.mode == 'w':
            self.conn.commit()
        self.conn.close()

    def _load_image(self, image_entry):
        logging.debug('Reading image "%s"' %
                      backendDb.imageField(image_entry, 'imagefile'))
        image = self.imreader.imread(
            backendDb.imageField(image_entry, 'imagefile'))
        if backendDb.imageField(image_entry, 'maskfile') is None:
            mask = None
        else:
            mask = self.imreader.maskread(
                backendDb.imageField(image_entry, 'maskfile'))
        return image, mask

    def __len__(self):
        return len(self.image_entries)

    def __getitem__(self, index):
        '''
        Used to train for the detection/segmentation task.
        Args:
            index:      An index of an image in the dataset.
        Returns a dict with keys:
            image:      np.uint8 array corresponding to an image.
            mask:       np.uint8 array corresponding to a mask if exists, or None.
            objects:    np.int32 array of shape [Nx5].
                        Each row is (x1, y1, width, height, classname)
            imagefile:  The image id.
        '''
        image_entry = self.image_entries[index]
        image, mask = self._load_image(image_entry)

        imagefile = backendDb.imageField(image_entry, 'imagefile')
        imagename = backendDb.imageField(image_entry, 'name')
        imagescore = backendDb.imageField(image_entry, 'score')

        keys = ['x1', 'y1', 'width', 'height', 'name', 'score']
        self.c.execute(
            'SELECT %s FROM objects WHERE imagefile=? AND (%s)' %
            (','.join(keys), self.where_object), (imagefile, ))
        object_entries = self.c.fetchall()
        # Convert to a list of dicts, each dict with the same keys.
        object_entries = [
            dict(zip(keys, object_entry)) for object_entry in object_entries
        ]

        sample = {
            'image': image,
            'mask': mask,
            'objects': object_entries,
            'imagefile': imagefile,
            'name': imagename,
            'score': imagescore,
        }

        sample = _filter_keys(self.used_keys, sample)
        sample = _apply_transform(self.transform_group, sample)
        return sample


class ObjectDataset(torch.utils.data.Dataset):
    ''' Items of a dataset are objects. '''
    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_object='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 used_keys=None,
                 transform_group=None):
        '''
        Args:
            db_file:        (string) A path to an sqlite3 database file.
            rootdir:        (string) A root path, is pre-appended to "imagefile"
                            entries of "images" table in the sqlite3 database.
            where_object:   (string) The WHERE part of the SQL query on the 
                            "objects" table, as in: 
                            "SELECT * FROM objects WHERE ${where_object};"
                            Allows to query only needed objects.
            mode:           ("r" or "w") The readonly or write-read mode to open 
                            the database. The default is "r", use "w" only if 
                            you plan to call addRecord().
            copy_to_memory: (bool) Copies database into memory. 
                            Only for mode="r". Should be used for python2. 
            used_keys:      (a list of strings or None) If specified, use only
                            these keys for every sample, and discard the rest.
            transform_group: ((1) a callable, or (2) a list of callables, 
                            or (3) a dict {string: callable}) 
                            Transform(s) to be applied on a sample.
                            (1) A callable: It is applied to the sample.
                            (2) A list of callables: Each callable is applied 
                                to the sample sequentially.
                            (3) A dict {string: callable}: Each key of this dict
                            should match a key in each sample, and the callables
                            are applied to the respective sample dict values.
        '''

        self.mode = mode
        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' %
                       where_object)
        self.object_entries = self.c.fetchall()

        self.imreader = backendMedia.MediaReader(rootdir=rootdir)

        self.used_keys = used_keys
        self.transform_group = transform_group

    def close(self):
        if self.mode == 'w':
            self.conn.commit()
        self.conn.close()

    def __len__(self):
        return len(self.object_entries)

    def __getitem__(self, index):
        '''
        Used to train for classification / segmentation of individual objects.
        Args:
            index:      (int) An index of an image in the dataset.
        Returns a dict with keys:
            image:      (np.uint8) The array corresponding to an image.
            mask:       (np.uint8) The array corresponding to a mask if exists, or None.
            objectid:   (int) The primary key in the "objects" table.
            name:       (string) The name field in the "objects" table.
            score:      (float) The score field in the "objects" table.
            imagefile:  (string) The image id.
            All key-value pairs from the "properties" table for this objectid.
        '''
        object_entry = self.object_entries[index]

        objectid = backendDb.objectField(object_entry, 'objectid')
        imagefile = backendDb.objectField(object_entry, 'imagefile')
        name = backendDb.objectField(object_entry, 'name')
        score = backendDb.objectField(object_entry, 'score')

        self.c.execute('SELECT maskfile FROM images WHERE imagefile=?',
                       (imagefile, ))
        maskfile = self.c.fetchone()[0]

        logging.debug('Reading object %d from %s imagefile' %
                      (objectid, imagefile))

        image = self.imreader.imread(imagefile)
        mask = self.imreader.maskread(
            maskfile) if maskfile is not None else None

        roi = backendDb.objectField(object_entry, 'roi')
        logging.debug('Roi: %s' % roi)
        image = image[roi[0]:roi[2], roi[1]:roi[3]]
        mask = mask[roi[0]:roi[2], roi[1]:roi[3]] if mask is not None else None

        sample = {
            'image': image,
            'mask': mask,
            'name': name,
            'score': score,
            'imagefile': imagefile,
            'objectid': objectid
        }

        # Add properties.
        self.c.execute('SELECT key,value FROM properties WHERE objectid=?',
                       (objectid, ))
        property_entries = self.c.fetchall()
        sample.update(dict(property_entries))

        sample = _filter_keys(self.used_keys, sample)
        sample = _apply_transform(self.transform_group, sample)
        return sample

    def addRecord(self, objectid, key, value):
        '''
        Add an entry to "properties" table.
        Used to add information to the database, e.g. during inference.
        Requires the Dataset be constructed with mode="w".
        This function is a thin wrapper around an "INSERT" SQL qeuery.

        Args:
            objectid: (int) An id of an object. 
                      It is a key in a dict returned by the [] operator. 
                      Tip: it will not work if you pass "used_keys" argument
                      without "objectid" when creating the dataset object.
            key:      (string) Describes what you are writing, e.g. "result".
            value:    (string or None) The value, converted to string.

        Returns:
            None
        '''
        if self.mode != 'w':
            raise ValueError(
                'This dataset was created in read-only mode, can not write. '
                'Please pass mode="w" when creating the Dataset.')

        self.c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
            (objectid, key, value))


####      Helper functions.    ####


def _filter_keys(used_keys, sample):
    '''
    Keep only `used_keys` in the dict.
    Args:
        used_keys:      (list of str) A list of keys to keep in the sample.
        sample:         (dict) A sample queried from one of the datasets.
    Returns:
        sample:         (dict) The sample sample but with filtered elements.
    '''
    if used_keys is not None:
        sample = {
            key: value
            for (key, value) in sample.items() if key in used_keys
        }
    return sample


def _apply_transform(transform_group, sample):
    ''' 
    Apply a group of transforms to a sample.
    Args:
        transform_group:  Can be one of the four options.
                        1. (None) Return `sample` unchanged. 
                        2. (callable) It is applied to the `sample` once.
                        3. (list of callables) Each callable is applied to the
                           sample. The results are put together in a list.
                           This option allows to apply one single transform on
                           image and mask together, and another transform on
                           e.g. name.
                        4. (dict string -> callable) Each key of this dict
                           should match a key in each sample, and the callables 
                           are applied to the respective sample dict values.
        sample:         (dict) A sample queried from one of the datasets.
    Returns:
        sample:         (any) Sample to further go into Pytorch TrainLoader.
    '''

    if transform_group is None:
        return sample

    elif callable(transform_group):
        return transform_group(sample)

    elif isinstance(transform_group, list):
        return [transform(sample) for transform in transform_group]

    elif isinstance(transform_group, dict):
        for key in transform_group:
            if key not in sample:
                raise KeyError('Key "%s" is in transform, but not in sample' %
                               key)
            sample[key] = transform_group[key](sample[key])
        return sample

    else:
        raise TypeError('Unexpected type of the tranform: %s' %
                        type(transform_group))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_db_file', required=True)
    parser.add_argument('--rootdir', required=True)
    parser.add_argument('--dataset_type',
                        required=True,
                        choices=['images', 'objects'])
    parser.add_argument('--mode', choices=['r', 'w'], default='r')
    args = parser.parse_args()

    if args.dataset_type == 'images':
        dataset = ImageDataset(args.in_db_file, rootdir=args.rootdir)
        item = dataset[1]
        pprint.pprint(item)

    elif args.dataset_type == 'objects':
        dataset = ObjectDataset(args.in_db_file, rootdir=args.rootdir)
        item = dataset[1]
        pprint.pprint(item)
