import logging
import torch.utils.data

from shuffler.backend import backend_media
from shuffler.interface import utils


class ImageDataset(torch.utils.data.Dataset):
    '''
    Items of a dataset are images, each one with its objects.
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
            used_keys:      (None or list of str)
                            Originally __getitem__ returns a dict with keys:
                            'image', 'mask', 'objects', 'imagefile', 'name',
                            'score' (see the comments to this class above).
                            Argument `used_keys` determines which of these keys
                            are needed, and which can be disposed of.
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
        utils.checkWhereImage(where_image)
        self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' %
                       where_image)
        self.image_entries = self.c.fetchall()

        self.imreader = backend_media.MediaReader(rootdir=rootdir)

        utils.checkWhereObject(where_object)
        self.where_object = where_object

        _checkUsedKeys(used_keys)
        self.used_keys = used_keys
        utils.checkTransformGroup(transform_group)
        self.transform_group = transform_group

    def close(self):
        if self.mode == 'w':
            self.conn.commit()
        self.conn.close()

    def execute(self, *args, **kwargs):
        ''' A thin wrapper that conceals self.c and its methods. '''

        self.c.execute(*args, **kwargs)
        return self.c.fetchall()

    # TODO: deprecate in favor of 'execute'.
    @property
    def cursor(self):
        return self.s

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
        sample = utils.buildImageSample(image_entry, self.c, self.imreader,
                                        self.where_object)
        if sample is None:
            logging.warning('Returning None for index %d', index)
            return None
        sample = _filterKeys(self.used_keys, sample)
        sample = utils.applyTransformGroup(self.transform_group, sample)
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
                 transform_group=None,
                 preload_samples=False):
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
            preload_samples:  (bool) If true, will try to preload all samples
                            (including images) into memory in __init__.
        '''

        self.mode = mode
        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        utils.checkWhereObject(where_object)
        self.c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' %
                       where_object)
        self.object_entries = self.c.fetchall()

        self.imreader = backend_media.MediaReader(rootdir=rootdir)

        _checkUsedKeys(used_keys)
        self.used_keys = used_keys
        utils.checkTransformGroup(transform_group)
        self.transform_group = transform_group

        if not preload_samples:
            self.preloaded_samples = None
        else:
            self.preloaded_samples = []
            logging.info('Loading samples...')
            for index in range(len(self)):
                object_entry = self.object_entries[index]
                sample = utils.buildObjectSample(object_entry, self.c,
                                                 self.imreader)
                if sample is None:
                    logging.warning('Skip bad sample %d', index)
                else:
                    self.preloaded_samples.append(sample)
            logging.info('Loaded %d samples.', len(self))

    def close(self):
        if self.mode == 'w':
            self.conn.commit()
        self.conn.close()

    def execute(self, *args, **kwargs):
        ''' A thin wrapper that conceals self.c and its methods. '''

        self.c.execute(*args, **kwargs)
        return self.c.fetchall()

    # TODO: deprecate in favor of 'execute'.
    @property
    def cursor(self):
        return self.s

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
        if self.preloaded_samples is not None:
            sample = self.preloaded_samples[index]
        else:
            logging.debug('Querying for %d-th object out of %d', index,
                          len(self.object_entries))
            object_entry = self.object_entries[index]
            sample = utils.buildObjectSample(object_entry, self.c,
                                             self.imreader)
            if sample is None:
                logging.warning('Returning None for index %d', index)
                return None

        sample = _filterKeys(self.used_keys, sample)
        sample = utils.applyTransformGroup(self.transform_group, sample)
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


def _checkUsedKeys(used_keys):
    if used_keys is None:
        return
    if not isinstance(used_keys, list):
        raise TypeError('used_keys is not list, but %s.' % type(used_keys))
    for key in used_keys:
        if not isinstance(key, str):
            raise TypeError('key "%s" of used_keys is not str, but %s.' %
                            (key, type(key)))


def _filterKeys(used_keys, sample):
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
