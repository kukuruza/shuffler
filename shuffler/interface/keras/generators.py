import os.path as op
import numpy as np
import tensorflow as tf

from shuffler.interface import utils
from shuffler.backend import backend_media


class ImageGenerator(tf.keras.utils.Sequence):
    '''
    Generates images, each one with objects.

    Every generated sample is a dict with the following fields:
        image:      (np.uint8 array) Corresponding to an image.
        mask:       (np.uint8 array) Corresponding to a mask if exists, or None.
        objects:    (np.int32 array of shape [Nx5])
                    Each row is (x1, y1, width, height, classname)
        imagefile:  (str) The "imagefile" field of the "images" table.
        name:       (str) The "name" field of the "images" table.
        score:      (float) The "score" field of the "images" table.

    You can query only a subset of these keys, and return them as a list, as
    a tuple, or as a dict (if a dict, you can change keys names) by providing
    `used_keys` to the __init__ (see comments to __init__.)
    '''

    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_image='TRUE',
                 where_object='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 used_keys=None,
                 transform_group=None,
                 batch_size=1,
                 shuffle=False):
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
            used_keys:      (None, list of str, tuple of str, or dict str -> str)
                            Originally __getitem__ returns a dict with keys:
                            'image', 'mask', 'objects', 'imagefile', 'name',
                            'score' (see the comments to this class above).
                            Argument `used_keys` determines which of these keys
                            are needed, and which can be disposed of.
                            Options for `used_keys`.
                            1) None. Each sample is unchanged dict.
                            2) List of str. Each str is a key.
                               __getitem__ returns a list.
                            3) Tuple of str. Same as above.
                               __getitem__ returns a tuple.
                            4) Dict str -> str. The key is the key in the
                               database, the value is the key in the output dict.
                               __getitem__ returns a dict.
            transform_group: (a callable or a dict string -> callable)
                            Transform(s) to be applied on a sample.
                            1) If it is a callable, it is applied to the sample.
                            2) If it is a dict, each key is matched to a key in
                               the sample (after used dict), and callables are
                               called on the respective elements.
        '''
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not op.exists(db_file):
            raise ValueError('db_file does not exist: %s' % db_file)

        self.mode = mode
        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM images WHERE %s ORDER BY imagefile' %
                       where_image)
        self.image_entries = self.c.fetchall()

        self.imreader = backend_media.MediaReader(rootdir=rootdir)
        self.where_object = where_object

        _checkUsedKeys(used_keys)
        self.used_keys = used_keys
        utils.checkTransformGroup(transform_group)
        self.transform_group = transform_group

        self.on_epoch_end()

    def close(self):
        ''' Crucial when the object is contructed in the 'w' mode. '''
        self.conn.close()

    def execute(self, *args, **kwargs):
        ''' A thin wrapper that conceals self.c and its methods. '''

        self.c.execute(*args, **kwargs)
        return self.c.fetchall()

    # Deprecate in favor of execute.
    @property
    def cursor(self):
        return self.c

    def __len__(self):
        ''' Denotes the number of batches per epoch. '''
        return int(np.ceil(len(self.image_entries) / self.batch_size))

    def _loadEntry(self, index):
        image_entry = self.image_entries[index]
        sample = utils.buildImageSample(image_entry, self.c, self.imreader,
                                        self.where_object)
        sample = _filterKeys(self.used_keys, sample)
        sample = utils.applyTransformGroup(self.transform_group, sample)
        return sample

    def __getitem__(self, index):
        '''
        Generate a batch of samples.
        Args:
            index:      (int) An index of the batch.
        Returns a dict with keys:
            batch:      (list of lists, tuple of lists, or dict with lists)
                        each element of the list / tuple / dict is one field
                        of the data, such as image or name.
                        you can choose between list / tuple / dict by providing
                        `used_keys` in the __init__.
        '''
        # Generate indexes of the batch.
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        samples = [self._loadEntry(index) for index in indexes]
        # Get a list of samples, each sample is a list, a tuple, or a dict.
        return _listOfWhateverToWhateverOfLists(samples)

    def on_epoch_end(self):
        ''' Updates indices after each epoch. '''

        self.indexes = np.arange(len(self.image_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class BareImageGenerator(ImageGenerator):
    ''' A specialization class providing only image and imagefile. '''

    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_image='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 transform_group=None,
                 batch_size=1,
                 shuffle=True):
        super(BareImageGenerator,
              self).__init__(db_file=db_file,
                             rootdir=rootdir,
                             where_image=where_image,
                             where_object='FALSE',
                             mode=mode,
                             copy_to_memory=copy_to_memory,
                             used_keys=['image', 'imagefile'],
                             transform_group=transform_group,
                             batch_size=batch_size,
                             shuffle=shuffle)


class ObjectGenerator(tf.keras.utils.Sequence):
    '''
    Generates objects, including cropped image and mask.

    Every generated sample is a dict with the following fields:
        image:      (np.uint8) The array corresponding to an image.
        mask:       (np.uint8) The array corresponding to a mask if exists, or None.
        objectid:   (int) The primary key in the "objects" table.
        name:       (string) The "name" field in the "objects" table.
        score:      (float) The "score" field in the "objects" table.
        imagefile:  (string) The image id.

    You can query only a subset of these keys, and return them as a list, as
    a tuple, or as a dict (if a dict, you can change keys names) by providing
    `used_keys` to the __init__ (see comments to __init__.)
    '''

    def __init__(self,
                 db_file,
                 rootdir='.',
                 where_object='TRUE',
                 mode='r',
                 copy_to_memory=True,
                 batch_size=1,
                 used_keys=None,
                 transform_group=None,
                 shuffle=False):
        '''
        Args:
            db_file:        (string) A path to an sqlite3 database file.
            rootdir:        (string) A root path, is pre-appended to "imagefile"
                            entries of "images" table in the sqlite3 database.
            where_object:   (string) The WHERE part of the SQL query on the
                            "objects" table, as in:
                            "SELECT * FROM objects WHERE ${where_object};"
                            Allows to query only needed objects for each image.
            mode:           ("r" or "w") The readonly or write-read mode to open
                            the database. The default is "r", use "w" only if
                            you plan to call addRecord().
            copy_to_memory: (bool) Copies database into memory.
                            Only for mode="r". Should be used for python2.
            used_keys:      (None, list of str, tuple of str, or dict str -> str)
                            Originally __getitem__ returns a dict with keys:
                            'image', 'mask', 'objects', 'imagefile', 'name',
                            'score' (see the comments to this class above).
                            Argument `used_keys` determines which of these keys
                            are needed, and which can be disposed of.
                            Options for `used_keys`.
                            1) None. Each sample is unchanged dict.
                            2) List of str. Each str is a key.
                               __getitem__ returns a list.
                            3) Tuple of str. Same as above.
                               __getitem__ returns a tuple.
                            4) Dict str -> str. The key is the key in the
                               database, the value is the key in the output dict.
                               __getitem__ returns a dict.
            transform_group: (a callable or a dict string -> callable)
                            Transform(s) to be applied on a sample.
                            1) If it is a callable, it is applied to the sample.
                            2) If it is a dict, each key is matched to a key in
                               the sample (after used dict), and callables are
                               called on the respective elements.
        '''

        self.batch_size = batch_size
        self.shuffle = shuffle

        if not op.exists(db_file):
            raise ValueError('db_file does not exist: %s' % db_file)

        self.mode = mode
        self.conn = utils.openConnection(db_file, mode, copy_to_memory)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' %
                       where_object)
        self.object_entries = self.c.fetchall()

        self.imreader = backend_media.MediaReader(rootdir=rootdir)

        _checkUsedKeys(used_keys)
        self.used_keys = used_keys
        utils.checkTransformGroup(transform_group)
        self.transform_group = transform_group

        self.on_epoch_end()

    def close(self):
        self.conn.close()

    def execute(self, *args, **kwargs):
        ''' A thin wrapper that conceals self.c and its methods. '''

        self.c.execute(*args, **kwargs)
        return self.c.fetchall()

    # Deprecate in favor of execute.
    @property
    def cursor(self):
        return self.c

    def __len__(self):
        ''' Denotes the number of batches per epoch. '''
        return int(np.ceil(len(self.object_entries) / self.batch_size))

    def _loadEntry(self, index):
        object_entry = self.object_entries[index]
        sample = utils.buildObjectSample(object_entry, self.c, self.imreader)
        sample = _filterKeys(self.used_keys, sample)
        sample = utils.applyTransformGroup(self.transform_group, sample)
        return sample

    def __getitem__(self, index):
        '''
        Generate a batch of samples.
        Args:
            index:      (int) An index of the batch.
        Returns a dict with keys:
            batch:      (list of lists, tuple of lists, or dict with lists)
                        each element of the list / tuple / dict is one field
                        of the data, such as image or name.
                        Choose between list / tuple / dict by specifying
                        used_keys when contructing the generator.
        '''
        # Generate indexes of the batch.
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        samples = [self._loadEntry(index) for index in indexes]
        # Get a list of samples, each sample is a list, a tuple, or a dict.
        return _listOfWhateverToWhateverOfLists(samples)

    def on_epoch_end(self):
        ''' Updates indexes after each epoch. '''
        self.indexes = np.arange(len(self.object_entries))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def addRecord(self, objectid, key, value):
        '''
        Add an entry to "properties" table.
        Used to add information to the database, e.g. during inference.
        Requires the Generator be constructed with mode="w".
        This function is a thin wrapper around an "INSERT" SQL qeuery.

        Args:
            objectid: (int) An id of an object.
                      It is a key in a dict returned by the [] operator.
                      Tip: it will not work if you pass "used_keys" argument
                      without "objectid" when creating the generator object.
            key:      (string) Describes what you are writing, e.g. "result".
            value:    (string or None) The value, converted to string.

        Returns:
            None
        '''
        if self.mode != 'w':
            raise ValueError(
                'This generator was created in read-only mode, can not write. '
                'Please pass mode="w" when creating the Generator.')

        self.c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
            (objectid, key, value))


####      Helper functions.    ####


def _listOfWhateverToWhateverOfLists(samples):

    def _transposeListOfLists(samples):
        return [[samples[j][i] for j in range(len(samples))]
                for i in range(len(samples[0]))]

    # A list of whatever to whatever of lists.
    if len(samples) == 0:
        return samples
    elif isinstance(samples[0], list):
        return _transposeListOfLists(samples)
    elif isinstance(samples[0], tuple):
        return tuple(_transposeListOfLists(samples))
    elif isinstance(samples[0], dict):
        return {k: [sample[k] for sample in samples] for k in samples[0]}


def _checkUsedKeys(used_keys):
    if used_keys is None:
        return
    elif isinstance(used_keys, list):
        for key in used_keys:
            if not isinstance(key, str):
                raise TypeError('key "%s" of used_keys is not str, but %s.' %
                                type(key))
    elif isinstance(used_keys, dict):
        for key in used_keys:
            if not isinstance(key, str):
                raise TypeError('key "%s" of used_keys is not str, but %s.' %
                                type(key))
            if not isinstance(used_keys[key], str):
                raise TypeError(
                    'value "%s" of key "%s" from used_keys is not str, but %s.'
                    % (key, used_keys[key], type(key)))
    else:
        raise TypeError('used_keys is not list, but %s.' % type(used_keys))


def _filterKeys(used_keys, sample):
    '''
    Keep only `used_keys` and return them as list, tuple, or dict.
    Args:
        used_keys:      (list of str, a tuple of str, a dict str -> str)
                        Determines fields captured in each sample, and the type
                        in which they are returned by __getitem__. Options:
                        1) List of str. Each str is a key. Return a list.
                        2) Tuple of str. Same as list, returns a tuple.
                        3) Dict str -> str. The key is the key in the database,
                           the value is the key in the output dict.
                           Return a dict.
        sample:         (dict) A sample queried from one of the datasets.
    Returns:
        sample:         (dict) The sample sample but with filtered elements.
    '''
    if used_keys is None:
        return sample

    elif isinstance(used_keys, list):
        return [sample[key] for key in used_keys]

    elif isinstance(used_keys, tuple):
        return tuple([sample[key] for key in used_keys])

    elif isinstance(used_keys, dict):
        return {value: sample[key] for (key, value) in used_keys.items()}

    else:
        raise TypeError('used_keys are not list, tuple, or dict: %s' %
                        type(used_keys))
