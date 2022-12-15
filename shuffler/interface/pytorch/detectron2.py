import logging
import functools
import detectron2.data

from shuffler.backend import backend_media
from shuffler.interface import utils


def register_object_dataset(dataset_name, *args, **kwargs):
    ''' This function registers a dataset in Detectron2 based on Shuffler db.
    Args:
        dataset_name:   (string) Any name, e.g. 'shuffler_object_dataset'.
        db_file:        (string) A path to an sqlite3 database file.
        rootdir:        (string) A root path, is pre-appended to "imagefile"
                        entries of "images" table in the sqlite3 database.
        where_object:   (string) The WHERE part of the SQL query on the
                        "objects" table, as in:
                        "SELECT * FROM objects WHERE ${where_object};"
                        Allows to query only needed objects.
        used_keys:      (a list of strings or None) If specified, use only
                        these keys for every sample, and discard the rest.
        transform_group: A dict {string: callable}. Each key of this dict
                        should match a key in each sample, and the callables
                        are applied to the respective sample dict values.
    '''
    detectron2.data.DatasetCatalog.register(
        dataset_name,
        functools.partial(_object_dataset_function, *args, **kwargs))


def _object_dataset_function(db_file,
                             rootdir='.',
                             where_object='TRUE',
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
        used_keys:      (a list of strings or None) If specified, use only
                        these keys for every sample, and discard the rest.
        transform_group: A dict {string: callable}. Each key of this dict
                        should match a key in each sample, and the callables
                        are applied to the respective sample dict values.
    '''

    conn = utils.openConnection(db_file, 'r', copy_to_memory=False)
    c = conn.cursor()
    utils.checkWhereObject(where_object)
    c.execute('SELECT * FROM objects WHERE %s ORDER BY objectid' %
              where_object)
    object_entries = c.fetchall()

    imreader = backend_media.MediaReader(rootdir=rootdir)

    _checkUsedKeys(used_keys)
    utils.checkTransformGroup(transform_group)

    samples = []
    logging.info('Loading samples...')
    for index in range(len(object_entries)):
        object_entry = object_entries[index]
        sample = utils.buildObjectSample(object_entry, c, imreader)
        if sample is None:
            continue
        sample = _filterKeys(used_keys, sample)
        sample = utils.applyTransformGroup(transform_group, sample)
        samples.append(sample)
    logging.info('Loaded %d samples.', len(object_entries))
    return samples


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
