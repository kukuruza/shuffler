import os, sys, os.path as op
import numpy as np
import logging
from pprint import pformat
import sqlite3
import argparse
from datetime import datetime
from collections import Mapping  # For checking argument type.

from lib.backend.backendDb import createDb, makeTimeString
from lib.backend.backendMedia import MediaWriter, getPictureSize


class DatasetWriter:
    '''
    Write a new dataset in Shuffler format, image by image, object by object.
    The database file (.db) and pictures or videos will be recorded.

    Images and masks can be provided either as numpy arrays (in that case, they
    are written on disk and the new path is recorded into the database), or as
    a path to an image (in that case, that path is recorded into the database)
    '''
    def __init__(self,
                 out_db_file,
                 rootdir='.',
                 media='pictures',
                 image_path=None,
                 mask_path=None,
                 overwrite=False):

        if out_db_file is None:
            raise TypeError('out_db_file is None')

        self.rootdir = rootdir
        self.media = media

        # Image and mask writer.
        self.imwriter = MediaWriter(media_type=media,
                                    rootdir=rootdir,
                                    image_media=image_path,
                                    mask_media=mask_path,
                                    overwrite=overwrite)

        # Maybe create directory for out_db_file.
        out_db_dir = op.dirname(out_db_file)
        if not op.exists(out_db_dir):
            os.makedirs(out_db_dir)

        # Create and open a database.
        if op.exists(out_db_file):
            if overwrite:
                os.remove(out_db_file)
            else:
                raise ValueError('"%s" already exists.' % out_db_file)
        self.conn = sqlite3.connect(out_db_file)
        self.c = self.conn.cursor()
        createDb(self.conn)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _checkValueIsNoneOrType(self, value, type_, name):
        if value is not None and not isinstance(value, type_):
            raise TypeError('%s should be of type %s, got %s' %
                            (name, str(type_), type(value)))

    def addImage(self, image_dict):
        '''
        Add the image path to the database and (maybe) write an image to disk.
        Args:
          image_dict:   A dict that should have some of the following keys:
                        * "image", "imagefile" (one of the two is required.)

                          "image" is a numpy array standing for an image.
                          If specified, the image will be recorded on disk as
                          a file or a video frame based on "media" argument of
                          the constructor. It will also be written as an entry
                          to "images" table.
                          If not specified, we consider that the image already
                          exists on disk.

                          "imagefile" is a path of the image on disk.
                          If specified, we consider that the image already
                          exists on disk, and this path will be recorded as an
                          entry to the "images" table.
                          If not specified, it is generated automoatically.

                        * "mask" or "maskfile" (at most one of the two.)

                          Same behavior as for "image" or "imagefile".
                          If skipped, no maskfile is written to "images" table.

                        * "timestamp" is an object of datetime.datetime type,
                          defines the time the image was created. If skipped,
                          the "now" time is recorded.

                        * "name" is a name/class of the image.

                        * "height", "width" define image dimensions in case
                          images are passed through imagefile/maskfile and were
                          recorded as video frames (media = "video")
        Returns:
          imagefile:    The path of the image relative to "rootdir"
        '''

        if not isinstance(image_dict, Mapping):
            raise TypeError('image_dict should be a dict, not %s' %
                            type(image_dict))

        ### Writing image.

        if ('image' in image_dict) == ('imagefile' in image_dict):
            raise ValueError(
                'Exactly one of "image" or "imagefile" must be specified.')

        if 'image' in image_dict:
            image = image_dict['image']
            # TODO: add namehint, so the behavior will change.
            imagefile = self.imwriter.imwrite(image)
            height, width = image.shape[0:2]

        elif 'imagefile' in image_dict:
            imagefile = image_dict['imagefile']

            # Check image on disk, and get its dimensions.
            if self.media == 'pictures':
                imagepath = op.join(self.rootdir, imagefile)
                if not op.exists(imagepath):
                    raise ValueError('Got imagefile "%s" with rootdir "%s", '
                                     'but cant find an image at "%s"' %
                                     (imagefile, self.rootdir, imagepath))
                height, width = getPictureSize(imagepath)
            # Check video on disk, and get the dimensions from user's input.
            elif self.media == 'video':
                videopath = op.join(self.rootdir, op.dirname(imagefile))
                if not op.exists(videopath):
                    raise ValueError('Got imagefile "%s" with rootdir "%s", '
                                     'but cant find the video at "%s"' %
                                     (imagefile, self.rootdir, videopath))
                if 'height' not in image_dict or 'width' not in image_dict:
                    # TODO: get the dimensions once for every video and store them,
                    #       instead of asking user to provide it. Do it only if anybody
                    #       is interested in this scenario.
                    raise KeyError(
                        '"height" and "width" should be specified in image_dict.'
                    )
                height = image_dict['height']
                width = image_dict['width']
                self._checkValueIsNoneOrType(value=height,
                                             type_=int,
                                             name='height')
                self._checkValueIsNoneOrType(value=width,
                                             type_=int,
                                             name='width')
            else:
                assert False  # No other type of "media" for now.

        else:
            assert False  # Either "image" or "imagefile" must be provided.

        ### Writing mask.

        if 'mask' in image_dict and 'maskfile' in image_dict:
            raise ValueError(
                'Only one of "mask" or "maskfile" can be specified.')

        if 'mask' in image_dict:
            mask = image_dict['mask']
            maskfile = self.imwriter.maskwrite(mask)

        elif 'maskfile' in image_dict:
            maskfile = image_dict['maskfile']

            # Check image on disk, and get its dimensions.
            if self.media == 'pictures':
                maskpath = op.join(self.rootdir, maskfile)
                if not op.exists(maskpath):
                    raise ValueError('Got maskfile "%s" with rootdir "%s", '
                                     'but cant find an image at "%s"' %
                                     (maskfile, self.rootdir, maskpath))
            # Check video on disk, and get the dimensions from user's input.
            elif self.media == 'video':
                videopath = op.join(self.rootdir, op.dirname(maskfile))
                if not op.exists(videopath):
                    raise ValueError('Got maskfile "%s" with rootdir "%s", '
                                     'but cant find the video at "%s"' %
                                     (maskfile, self.rootdir, videopath))
            else:
                assert False  # No other type of "media" for now.

        else:
            maskfile = None

        ### Timestamp.

        if 'timestamp' in image_dict:
            timestamp = image_dict['timestamp']
            if not isinstance(timestamp, datetime):
                raise TypeError(
                    'Timestamp should be an object of datetime.datetime class. '
                    'Instead got %s' % timestamp)
        else:
            timestamp = makeTimeString(datetime.now())

        ### Name and score.

        name = image_dict['name'] if 'name' in image_dict else None
        score = image_dict['score'] if 'score' in image_dict else None

        ### Record.

        entry = (imagefile, width, height, maskfile, timestamp, name, score)
        s = 'images(imagefile,width,height,maskfile,timestamp,name,score)'
        logging.debug('Writing %s to %s', entry, s)
        self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?)' % s, entry)
        return imagefile

    def addObject(self, object_dict):
        '''
        Record an object into the database.
        Args:
          object_dict:  A dict that should have some of the following keys:

                        * "objectid" (optional) defines the id of the object
                          in the "objects" table. When omitted, a new unique
                          objectid will be created.
                          Omit it, unless you have good reasons to use it.

                        * "imagefile" defines an entry in the "images" table
                          that this object belongs to.

                        * "x1", "y1", "width", "height" define the bounding box.

                        * "name" defines the name/class.

                        * "score" defines the score/confidence.

                        Other dict entries are interpreted as object properties,
                        and are recorded into the "properties" table.
                        Recording object polygons is NOT implemented now.

        Returns:
          objectid:     The id of the object in the "objects" table.
        '''

        if not isinstance(object_dict, Mapping):
            raise TypeError('object_dict should be a dict, not %s' %
                            type(object_dict))

        if 'imagefile' not in object_dict:
            raise KeyError('"imagefile" is required in object_dict, got %s' %
                           object_dict)

        imagefile = object_dict['imagefile']
        x1 = object_dict['x1'] if 'x1' in object_dict else None
        y1 = object_dict['y1'] if 'y1' in object_dict else None
        width = object_dict['width'] if 'width' in object_dict else None
        height = object_dict['height'] if 'height' in object_dict else None
        name = object_dict['name'] if 'name' in object_dict else None
        score = object_dict['score'] if 'score' in object_dict else None

        self._checkValueIsNoneOrType(value=width, type_=int, name='width')
        self._checkValueIsNoneOrType(value=height, type_=int, name='height')
        self._checkValueIsNoneOrType(value=x1, type_=int, name='x1')
        self._checkValueIsNoneOrType(value=y1, type_=int, name='y1')
        self._checkValueIsNoneOrType(value=score, type_=float, name='score')

        # Insert values into "objects" table.
        if 'objectid' in object_dict:
            objectid = object_dict['objectid']
            object_entry = (objectid, imagefile, x1, y1, width, height, name,
                            score)
            s = 'objects(objectid,imagefile,x1,y1,width,height,name,score)'
            logging.debug('Writing %s to %s (objectid is provided.)' %
                          (object_entry, s))
            self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?,?)' % s,
                           object_entry)
        else:
            object_entry = (imagefile, x1, y1, width, height, name, score)
            s = 'objects(imagefile,x1,y1,width,height,name,score)'
            logging.debug('Writing %s to %s (objectid not provided.)' %
                          (object_entry, s))
            self.c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?)' % s,
                           object_entry)
            objectid = self.c.lastrowid

        # Insert values into "properties" table.
        for key in object_dict:
            if key not in [
                    'objectid', 'imagefile', 'x1', 'y1', 'width', 'height',
                    'name', 'score'
            ]:
                property_entry = (objectid, key, str(object_dict[key]))
                s = 'properties(objectid,key,value)'
                logging.debug('Writing %s to %s' % (property_entry, s))
                self.c.execute('INSERT INTO %s VALUES (?,?,?)' % s,
                               property_entry)

        return objectid

    def addMatch(self, objectid, match=None):
        '''
        Record a group of matched objects in the database.

        First call this function with one object and grab the returned "match",
        then pass that match when calling function for every new matched object.
        Adding each object to the match requires calling the function once.

        Example:
        ```
        match1 = myDatasetWriter.addMatch (objectid1)
        myDatasetWriter.addMatch (objectid2, match=match1)

        match2 = myDatasetWriter.addMatch (objectid3)
        myDatasetWriter.addMatch (objectid4, match=match2)
        myDatasetWriter.addMatch (objectid5, match=match2)
        ```

        Args:
          objectid:   The id of one of the objects to match.

          match:      Should be None, when adding the first object of a match.
                      Should an existing match, when matching another object.

        Returns:
          match:      The id of the match. Pass it as an argument when calling
                      the function for the second, third, etc time.
        '''
        if match is None:
            self.c.execute('SELECT MAX(match) FROM matches')
            match = self.c.fetchone()[0]
            match = match + 1 if match is not None else 0
        s = 'matches(match,objectid)'
        logging.debug('Adding a new match %d for objectid %d' %
                      (match, objectid))
        self.c.execute('INSERT INTO %s VALUES (?,?);' % s, (match, objectid))
        return match

    def close(self):
        self.imwriter.close()
        self.conn.commit()
        self.c.execute('SELECT COUNT(1) FROM images')
        logging.info('Wrote the total of %d entries to images.' %
                     self.c.fetchone()[0])
        self.conn.close()


if __name__ == "__main__":
    '''
    Demo of how to use DatasetWriter.
    Writes images and objects with some dummy data.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_db_file', required=True)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        default='pictures',
        help='Whether to write images as JPG/PNG files or as frames in a video.'
    )
    parser.add_argument(
        '--image_path',
        help=
        'If media is "pictures", pass the path to the new folder with pictures. '
        'If media is "video", pass the path to that video.')
    parser.add_argument(
        '--mask_path',
        help=
        'If media is "pictures", pass the path to the new folder with pictures. '
        'If media is "video", pass the path to that video.')
    parser.add_argument(
        '--rootdir',
        default='.',
        help=
        'Directory that "imagefile" and "maskfile" entries will be relative to.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recreate images and the database, if they already exist.')
    args = parser.parse_args()

    # Create an instance of DatasetWriter
    with DatasetWriter(
            out_db_file=args.out_db_file,
            media=args.media,
            image_path=args.image_path,
            mask_path=args.mask_path,
            rootdir=args.rootdir,
            overwrite=args.overwrite,
    ) as writer:

        # Add two images.
        imagefile1 = writer.addImage({
            'image':
            np.random.randint(255, size=(100, 100, 3), dtype=np.uint8),
            'mask':
            np.random.randint(128, size=(100, 100), dtype=np.uint8),
        })
        imagefile2 = writer.addImage({
            'image':
            np.random.randint(255, size=(100, 100, 3), dtype=np.uint8),
        })

        # Add two objects.
        objectid1 = writer.addObject({
            'imagefile': imagefile1,
            'x1': 10,
            'y1': 20,
            'width': 30,
            'height': 40,
            'name': 'myobject',
            'fancy color property': 'green',
        })
        objectid2 = writer.addObject({
            'imagefile': imagefile2,
            'x1': 40,
            'y1': 30,
            'width': 20,
            'height': 10,
            'name': 'myobject',
        })

        # Add one match.
        match1 = writer.addMatch(objectid=objectid1)
        writer.addMatch(objectid=objectid2, match=match1)

    # Check what we have written
    conn = sqlite3.connect(args.out_db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM images')
    print('Images:\n', pformat(cursor.fetchall()))
    cursor.execute('SELECT * FROM objects')
    print('Objects:\n', pformat(cursor.fetchall()))
    cursor.execute('SELECT * FROM matches')
    print('Matches:\n', pformat(cursor.fetchall()))
    cursor.execute('SELECT * FROM properties')
    print('Properties:\n', pformat(cursor.fetchall()))
    conn.close()
