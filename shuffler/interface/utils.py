import sqlite3
import logging
import traceback

from shuffler.backend import backend_db


def openConnection(db_file, mode='r', copy_to_memory=True):
    '''
    Open a sqlite3 database connection.
      db_file:           Path to an sqlite3 database
      mode:              Can be 'r' or 'w'. If 'r', forbid writing.
      copy_to_memory:    A parameter only for read-only mode.
                         If yes, load the whole database to memory.
                         Otherwise, open as file:%s?mode=ro.
    '''
    if mode not in ['w', 'r']:
        raise ValueError('"mode" must be "w" or "r".')
    elif mode == 'w':
        conn = sqlite3.connect(db_file)
    elif copy_to_memory:
        conn = sqlite3.connect(':memory:')  # create a memory database
        disk_conn = sqlite3.connect(db_file)
        query = ''.join(line for line in disk_conn.iterdump())
        conn.executescript(query)
    else:
        try:
            conn = sqlite3.connect('file:%s?mode=ro' % db_file, uri=True)
        except TypeError:
            raise TypeError(
                'This Python version does not support connecting to SQLite by uri'
            )
    return conn


def buildImageSample(image_entry, cursor, imreader, where_object='TRUE'):
    '''
    Load images and get necessary information from object_entry to make a frame.
    Args:
        image_entry:  (tuple) All fields from "images" table.
        cursor:       (sqlite3 cursor)
        imreader:     (lib.backend.backendMedia.MediaReader)
        where_object: (str) condition to query the "objects" table.
    Returns a dict with keys:
        image:      (np.uint8 array) Corresponding to an image.
        mask:       (np.uint8 array) Corresponding to a mask if exists, or None.
        objects:    (np.int32 array of shape [Nx5])
                    Each row is (x1, y1, width, height, classname)
        imagefile:  (str) The "imagefile" field of the "images" table.
        name:       (str) The "name" field of the "images" table.
        score:      (float) The "score" field of the "images" table.
    '''
    imagefile = backend_db.imageField(image_entry, 'imagefile')
    maskfile = backend_db.imageField(image_entry, 'maskfile')
    image_width = backend_db.imageField(image_entry, 'width')
    image_height = backend_db.imageField(image_entry, 'height')

    # Load the image and mask.
    logging.debug('Reading image "%s"', imagefile)
    try:
        image = imreader.imread(imagefile)
        mask = imreader.maskread(maskfile) if maskfile is not None else None
    except ValueError:
        traceback.print_exc()
        logging.error('Reading image or mask failed. Returning None.')
        return None

    imagename = backend_db.imageField(image_entry, 'name')
    imagescore = backend_db.imageField(image_entry, 'score')

    keys = ['x1', 'y1', 'width', 'height', 'name', 'score']
    cursor.execute(
        'SELECT %s FROM objects WHERE imagefile=? AND (%s)' %
        (','.join(keys), where_object), (imagefile, ))
    object_entries = cursor.fetchall()
    # Convert to a list of dicts, each dict with the same keys.
    object_entries = [
        dict(zip(keys, object_entry)) for object_entry in object_entries
    ]

    sample = {
        'image': image,
        'mask': mask,
        'image_width': image_width,
        'image_height': image_height,
        'objects': object_entries,
        'imagefile': imagefile,
        'name': imagename,
        'score': imagescore,
    }
    return sample


def buildObjectSample(object_entry, c, imreader):
    '''
    Load images and get necessary information from object_entry to make a frame.
    Args:
        object_entry: (tuple) tuple with all fields from "objects" table.
        c:          (cursor) sqlite3 cursor.
        imreader:   (lib.backend.backendMedia.MediaReader)
    Returns a dict with keys:
        image:      (np.uint8) The array corresponding to an image.
        mask:       (np.uint8) The array corresponding to a mask if exists, or None.
        objectid:   (int) The primary key in the "objects" table.
        name:       (string) The "name" field in the "objects" table.
        score:      (float) The "score" field in the "objects" table.
        imagefile:  (string) The image id.
        All key-value pairs from the "properties" table for this objectid.
    '''
    objectid = backend_db.objectField(object_entry, 'objectid')
    imagefile = backend_db.objectField(object_entry, 'imagefile')
    name = backend_db.objectField(object_entry, 'name')
    score = backend_db.objectField(object_entry, 'score')

    c.execute('SELECT maskfile FROM images WHERE imagefile=?', (imagefile, ))
    maskfile = c.fetchone()[0]

    logging.debug('Reading object %d from %s imagefile', objectid, imagefile)

    try:
        image = imreader.imread(imagefile)
        mask = imreader.maskread(maskfile) if maskfile is not None else None
    except ValueError:
        traceback.print_exc()
        logging.error('Reading image or mask failed. Returning None.')
        return None

    roi = backend_db.objectField(object_entry, 'roi')
    logging.debug('Roi: %s', roi)
    roi = [int(x) for x in roi]
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
    c.execute('SELECT key,value FROM properties WHERE objectid=?',
              (objectid, ))
    property_entries = c.fetchall()
    sample.update(dict(property_entries))

    return sample


def applyTransformGroup(transform_group, sample):
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
                logging.debug('Key %s is not in sample. Will use None.', key)
                sample[key] = None
            sample[key] = transform_group[key](sample[key])
        return sample

    else:
        raise TypeError('Unexpected type of the tranform group: %s' %
                        type(transform_group))


def checkTransformGroup(transform_group):
    ''' Check the type of "transform_group", used in Pytorch and Keras. '''
    if transform_group is None or callable(transform_group):
        return
    elif isinstance(transform_group, list):
        for transform in transform_group:
            if not callable(transform):
                raise TypeError('Transform "%s" is not callable.' % transform)
    elif isinstance(transform_group, dict):
        for key in transform_group:
            if not callable(transform_group[key]):
                raise TypeError('Transform "%s" is not callable.' %
                                transform_group[key])
    else:
        raise TypeError('transform_group is not callable or dict, but %s.' %
                        type(transform_group))


def checkWhereImage(where_image):
    ''' Check the type of "where_image", used in Pytorch and Keras. '''
    if not isinstance(where_image, str):
        raise TypeError('where_image is not str, but %s.' % type(where_image))


def checkWhereObject(where_object):
    ''' Check the type of "where_object", used in Pytorch and Keras. '''
    if not isinstance(where_object, str):
        raise TypeError('where_object is not str, but %s.' %
                        type(where_object))
