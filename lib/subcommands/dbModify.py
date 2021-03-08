import os, sys, os.path as op
import numpy as np
import cv2
import logging
import sqlite3
import imageio
from math import ceil
import random
from glob import glob
from pprint import pformat
from datetime import datetime
from progressbar import progressbar
from ast import literal_eval

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util
from lib.utils import utilBoxes


def add_parsers(subparsers):
    bboxesToPolygonsParser(subparsers)
    polygonsToBboxesParser(subparsers)
    sqlParser(subparsers)
    addVideoParser(subparsers)
    addPicturesParser(subparsers)
    headImagesParser(subparsers)
    tailImagesParser(subparsers)
    randomNImagesParser(subparsers)
    expandObjectsParser(subparsers)
    moveMediaParser(subparsers)
    moveRootdirParser(subparsers)
    addDbParser(subparsers)
    subtractDbParser(subparsers)
    splitDbParser(subparsers)
    mergeIntersectingObjectsParser(subparsers)
    renameObjectsParser(subparsers)
    resizeAnnotationsParser(subparsers)
    propertyToNameParser(subparsers)
    syncObjectidsWithDbParser(subparsers)
    syncObjectsDataWithDbParser(subparsers)
    revertObjectTransformsParser(subparsers)


def bboxesToPolygonsParser(subparsers):
    parser = subparsers.add_parser(
        'bboxesToPolygons',
        description='If polygons dont exist for an object, '
        'create a rectangular polygon from the bounding box.')
    parser.set_defaults(func=bboxesToPolygons)


def bboxesToPolygons(c, args):
    c.execute('SELECT objectid FROM objects WHERE objectid NOT IN '
              '(SELECT objectid FROM polygons)')
    for objectid, in progressbar(c.fetchall()):
        util.bboxes2polygons(c, objectid)


def polygonsToBboxesParser(subparsers):
    parser = subparsers.add_parser(
        'polygonsToBboxes',
        description='Update bounding box in the "objects" table '
        'with values from "polygons".')
    parser.set_defaults(func=polygonsToBboxes)


def polygonsToBboxes(c, args):
    c.execute('SELECT objectid FROM objects WHERE objectid NOT IN '
              '(SELECT DISTINCT(objectid) FROM polygons)')
    for objectid, in progressbar(c.fetchall()):
        util.polygons2bboxes(c, objectid)


def sqlParser(subparsers):
    parser = subparsers.add_parser(
        'sql',
        description='Run SQL commands.'
        'Recorded paths will be made relative to "rootdir" argument.')
    parser.add_argument('--sql', nargs='+', help='A list of SQL statements.')
    parser.set_defaults(func=sql)


def sql(c, args):
    for command in args.sql:
        logging.info('Executing SQL command: %s', command)
        c.execute(command)


def addVideoParser(subparsers):
    parser = subparsers.add_parser(
        'addVideo',
        description='Import frames from a video into the database. '
        'Recorded paths will be made relative to "rootdir" argument.')
    parser.add_argument('--image_video_path', required=True)
    parser.add_argument('--mask_video_path')
    parser.set_defaults(func=addVideo)


def addVideo(c, args):

    # Check video paths.
    if not op.exists(args.image_video_path):
        raise FileNotFoundError('Image video does not exist at: %s' %
                                args.image_video_path)
    if args.mask_video_path is not None and not op.exists(
            args.mask_video_path):
        raise FileNotFoundError('Mask video does not exist at: %s' %
                                args.mask_video_path)

    # Check the length of image video with imageio
    image_video = imageio.get_reader(args.image_video_path)
    image_length = image_video.get_length()
    image = image_video.get_data(0)
    image_video.close()
    logging.info('Video has %d frames', image_length)
    if image_length == 0:
        raise ValueError('The image video is empty.')

    # Check if masks agree with images.
    if args.mask_video_path is not None:
        mask_video = imageio.get_reader(args.image_video_path)
        mask_length = mask_video.get_length()
        mask = mask_video.get_data(0)
        mask_video.close()
        if image_length != mask_length:
            raise ValueError('Image length %d mask video length %d mismatch' %
                             (image_length, mask_length))
        if image.shape[0:2] != mask.shape[0:2]:
            # The mismatch may happen if a mask is a CNN predicted image
            # of almost the same shape.
            logging.warning('Image size %s and mask size %s mismatch.',
                            image.shape[1], mask.shape[1])

    # Get the paths.
    image_video_rel_path = op.relpath(op.abspath(args.image_video_path),
                                      args.rootdir)
    if args.mask_video_path is not None:
        mask_video_rel_path = op.relpath(op.abspath(args.mask_video_path),
                                         args.rootdir)

    # Write to db.
    for iframe in progressbar(range(image_length)):
        height, width = image.shape[0:2]
        imagefile = op.join(image_video_rel_path, '%06d' % iframe)
        maskfile = op.join(mask_video_rel_path, '%06d' %
                           iframe) if args.mask_video_path else None
        timestamp = backendDb.makeTimeString(datetime.now())
        c.execute(
            'INSERT INTO images('
            'imagefile, width, height, maskfile, timestamp) VALUES (?,?,?,?,?)',
            (imagefile, width, height, maskfile, timestamp))


def addPicturesParser(subparsers):
    parser = subparsers.add_parser(
        'addPictures',
        description='Import picture files into the database.'
        'File names (without extentions) for images and masks must match.')
    parser.add_argument(
        '--image_pattern',
        required=True,
        help='Wildcard pattern for image files. E.g. "my/path/images-\\*.jpg"'
        'Escape "*" with quotes or backslash.')
    parser.add_argument(
        '--mask_pattern',
        default='/dummy',
        help='Wildcard pattern for image files. E.g. "my/path/masks-\\*.png"'
        'Escape "*" with quotes or backslash.')
    parser.add_argument(
        '--width_hint',
        type=int,
        help='If specified, the width is assumed to be the same across all '
        'images, and is not queried from individual images on disk.')
    parser.add_argument(
        '--height_hint',
        type=int,
        help='If specified, the height is assumed to be the same across all '
        'images, and is not queried from individual images on disk.')
    parser.set_defaults(func=addPictures)


def addPictures(c, args):

    # Collect a list of paths.
    image_paths = sorted(glob(args.image_pattern))
    logging.debug('image_paths:\n%s', pformat(image_paths, indent=2))
    if not image_paths:
        logging.error('Image files do not exist for the frame pattern: %s',
                      args.image_pattern)
        return
    mask_paths = sorted(glob(args.mask_pattern))
    logging.debug('mask_paths:\n%s', pformat(mask_paths, indent=2))

    def _nameWithoutExtension(x):
        '''
        File name without extension is the shared part between images and masks.
        '''
        return op.splitext(op.basename(x))[0]

    def _matchPaths(A, B):
        '''
        Groups paths in lists A and B into pairs, e.g. [(a1, b1), (a2, None), ...]
        '''
        A = {_nameWithoutExtension(path): path for path in A}
        B = {_nameWithoutExtension(path): path for path in B}
        return [(A[name], (B[name] if name in B else None)) for name in A]

    # Find correspondences between images and masks.
    pairs = _matchPaths(image_paths, mask_paths)
    logging.debug('Pairs:\n%s', pformat(pairs, indent=2))
    logging.info('Found %d images, %d of them have masks.', len(pairs),
                 len([x for x in pairs if x[1] is not None]))

    # Write to database.
    for image_path, mask_path in progressbar(pairs):
        if args.width_hint and args.height_hint:
            logging.debug('Using width %d and height %d from hint.',
                          args.width_hint, args.height_hint)
            height, width = args.height_hint, args.width_hint
        else:
            height, width = backendMedia.getPictureSize(image_path)
        imagefile = op.relpath(op.abspath(image_path), args.rootdir)
        maskfile = op.relpath(op.abspath(mask_path),
                              args.rootdir) if mask_path else None
        timestamp = backendDb.makeTimeString(datetime.now())
        c.execute(
            'INSERT INTO images('
            'imagefile, width, height, maskfile, timestamp) VALUES (?,?,?,?,?)',
            (imagefile, width, height, maskfile, timestamp))


def headImagesParser(subparsers):
    parser = subparsers.add_parser(
        'headImages', description='Keep the first N image entries.')
    parser.add_argument('-n', required=True, type=int)
    parser.set_defaults(func=headImages)


def headImages(c, args):
    c.execute('SELECT imagefile FROM images')
    imagefiles = c.fetchall()

    if len(imagefiles) < args.n:
        logging.info('Nothing to delete. Number of images is %d',
                     len(imagefiles))
        return

    for imagefile, in imagefiles[args.n:]:
        backendDb.deleteImage(c, imagefile)


def tailImagesParser(subparsers):
    parser = subparsers.add_parser(
        'tailImages', description='Keep the last N image entries.')
    parser.add_argument('-n', required=True, type=int)
    parser.set_defaults(func=tailImages)


def tailImages(c, args):
    c.execute('SELECT imagefile FROM images')
    imagefiles = c.fetchall()

    if len(imagefiles) < args.n:
        logging.info('Nothing to delete. Number of images is %d',
                     len(imagefiles))
        return

    for imagefile, in imagefiles[:-args.n]:
        backendDb.deleteImage(c, imagefile)


def randomNImagesParser(subparsers):
    parser = subparsers.add_parser('randomNImages',
                                   description='Keep random N image entries.')
    parser.add_argument('-n', required=True, type=int)
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.set_defaults(func=randomNImages)


def randomNImages(c, args):
    c.execute('SELECT imagefile FROM images')
    imagefiles = c.fetchall()
    random.seed(args.seed)
    random.shuffle(imagefiles)

    if len(imagefiles) < args.n:
        logging.info('Nothing to delete. Number of images is %d',
                     len(imagefiles))
        return

    for imagefile, in imagefiles[args.n:]:
        backendDb.deleteImage(c, imagefile)


def expandObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'expandObjects',
        description='Expand bbox and polygons in all four directions.')
    parser.set_defaults(func=expandObjects)
    parser.add_argument('--expand_perc', type=float, required=True)
    parser.add_argument(
        '--target_ratio',
        type=float,
        help='If specified, expand to match this height/width ratio, '
        'and if that is less than "expand_perc", then expand more.')


def expandObjects(c, args):
    c.execute('SELECT * FROM objects')
    object_entries = c.fetchall()
    logging.info('Found %d objects', len(object_entries))

    for object_entry in progressbar(object_entries):
        objectid = backendDb.objectField(object_entry, 'objectid')
        old_roi = backendDb.objectField(object_entry, 'roi')
        c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid, ))
        old_polygon = c.fetchall()

        # Scale.
        if args.target_ratio:
            if old_roi is not None:
                roi = utilBoxes.expandRoiToRatio(old_roi, args.expand_perc,
                                                 args.target_ratio)
                logging.debug('Roi changed from %s to %s for object %d',
                              str(old_roi), str(roi), objectid)
            if len(old_polygon):
                raise NotImplementedError(
                    'Cant scale polygons to target ratio. It is a TODO.')
        else:
            if old_roi is not None:
                roi = utilBoxes.expandRoi(old_roi,
                                          (args.expand_perc, args.expand_perc))
                logging.debug('Roi changed from %s to %s for object %d',
                              str(old_roi), str(roi), objectid)
            if len(old_polygon):
                ids = [backendDb.polygonField(p, 'id') for p in old_polygon]
                old_xs = [backendDb.polygonField(p, 'x') for p in old_polygon]
                old_ys = [backendDb.polygonField(p, 'y') for p in old_polygon]
                xs, ys = utilBoxes.expandPolygon(
                    old_xs, old_ys, (args.expand_perc, args.expand_perc))
                polygon = zip(ids, xs, ys)
                logging.debug('Polygon changed from %s to %s for object %d',
                              str(list(zip(old_xs, old_ys))),
                              str(list(zip(xs, ys))), objectid)

        # Update the database.
        if old_roi is not None:
            c.execute(
                'UPDATE objects SET x1=?, y1=?,width=?,height=? WHERE objectid=?',
                tuple(utilBoxes.roi2bbox(roi) + [objectid]))
        if len(old_polygon):
            for id, x, y in polygon:
                c.execute('UPDATE polygons SET x=?, y=? WHERE id=?',
                          (x, y, id))


def moveMediaParser(subparsers):
    parser = subparsers.add_parser(
        'moveMedia', description='Change imagefile and maskfile.')
    parser.set_defaults(func=moveMedia)
    parser.add_argument(
        '--where_image',
        default='TRUE',
        help='the SQL "where" clause for "images" table. '
        'For example, to change imagefile of JPG pictures from directory '
        '"from/mydir" only, use: \'imagefile LIKE "from/mydir/%%"\'')
    parser.add_argument(
        '--image_path',
        help='the directory for pictures OR video file of images')
    parser.add_argument(
        '--mask_path',
        help='the directory for pictures OR video file of masks')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        help='How many levels to keep in the directory structure. '
        'E.g. to move "my/old/fancy/image.jpg" to "his/new/fancy/image.jpg", '
        'specify image_path="his/new" and level=2. '
        'That will move subpath "fancy/image.jpg", i.e. of 2 levels.')
    parser.add_argument(
        '--adjust_size',
        action='store_true',
        help='Check the size of target media, and scale annotations')


def moveMedia(c, args):
    def splitall(path):
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path:  # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts

    def getPathBase(oldfile, level):
        old_path_parts = splitall(oldfile)
        if len(old_path_parts) < level:
            raise ValueError('Cant use "lavel"=%d on path "%s"' %
                             (level, oldfile))
        result = os.path.join(*old_path_parts[-args.level:])
        logging.debug('getPathBase produced "%s"', result)
        return result

    if args.image_path:
        logging.debug('Moving image dir to: %s', args.image_path)
        c.execute('SELECT imagefile FROM images WHERE (%s)' % args.where_image)
        imagefiles = c.fetchall()

        for oldfile, in progressbar(imagefiles):
            if oldfile is None:
                continue

            newfile = op.join(args.image_path,
                              getPathBase(oldfile, args.level))
            c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                      (newfile, oldfile))
            c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                      (newfile, oldfile))

            # TODO: this only works on images. Make for video.
            newpath = op.join(args.rootdir, newfile)
            if not op.exists(newpath):
                raise IOError('New file "%s" does not exist (rootdir "%s"), '
                              '(created from "%s")' %
                              (newpath, args.rootdir, oldfile))
            if args.adjust_size:
                c.execute('SELECT height, width FROM images WHERE imagefile=?',
                          (newfile, ))
                oldheight, oldwidth = c.fetchone()
                newheight, newwidth = backendMedia.getPictureSize(newpath)
                if newheight != oldheight or newwidth != oldwidth:
                    logging.info(
                        'Scaling annotations in "%s" from %dx%d to %dx%d.',
                        oldfile, oldheight, oldwidth, newheight, newwidth)
                    _resizeImageAnnotations(c, newfile, oldwidth, oldheight,
                                            newwidth, newheight)

    if args.mask_path:
        logging.debug('Moving mask dir to: %s', args.mask_path)
        c.execute('SELECT maskfile FROM images WHERE (%s)' % args.where_image)
        maskfiles = c.fetchall()

        for oldfile, in progressbar(maskfiles):
            if oldfile is not None:
                newfile = op.join(args.image_path,
                                  getPathBase(oldfile, args.level))
                c.execute('UPDATE images SET maskfile=? WHERE maskfile=?',
                          (newfile, oldfile))


def moveRootdirParser(subparsers):
    parser = subparsers.add_parser(
        'moveRootdir',
        description='Change imagefile and maskfile entries to be relative '
        'to the provided rootdir.')
    parser.set_defaults(func=moveRootdir)
    parser.add_argument('newrootdir',
                        help='All paths will be relative to the newrootdir.')


def _moveRootDir(c, oldrootdir, newrootdir):
    logging.info('Moving from rootdir %s to new rootdir %s', oldrootdir,
                 newrootdir)
    relpath = op.relpath(oldrootdir, newrootdir)
    logging.info('The path of oldroot relative to newrootdir is %s', relpath)

    c.execute('SELECT imagefile FROM images')
    for oldfile, in progressbar(c.fetchall()):
        if oldfile is not None:
            newfile = op.normpath(op.join(relpath, oldfile))
            c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                      (newfile, oldfile))
            c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                      (newfile, oldfile))

    c.execute('SELECT maskfile FROM images')
    for oldfile, in progressbar(c.fetchall()):
        if oldfile is not None:
            newfile = op.normpath(op.join(relpath, oldfile))
            c.execute('UPDATE images SET maskfile=? WHERE maskfile=?',
                      (newfile, oldfile))


def moveRootdir(c, args):
    _moveRootDir(c, args.rootdir, args.newrootdir)


def addDbParser(subparsers):
    parser = subparsers.add_parser(
        'addDb',
        description='Adds info from "db_file" to the current open database. '
        'Objects can be merged. Duplicate "imagefile" entries are ignore, '
        'but all associated objects are added.')
    parser.set_defaults(func=addDb)
    parser.add_argument('--db_file', required=True)
    parser.add_argument(
        '--db_rootdir',
        help=
        'If specified, imagefiles from add_db are considered relative to db_rootdir. '
        'They will be modified to be relative to rootdir of the active database.'
    )
    parser.add_argument('--merge_duplicate_objects', action='store_true')


def _maybeUpdateImageField(image_entry, image_entry_add, field):
    value_add = backendDb.imageField(image_entry_add, field)
    value = backendDb.imageField(image_entry, field)
    if value is None and value_add is not None:
        logging.debug('Field %s will set to %s.', field, str(value))
        image_entry = backendDb.setImageField(image_entry, field, value)
    return image_entry


def addDb(c, args):
    if not op.exists(args.db_file):
        raise FileNotFoundError('File does not exist: %s' % args.db_file)
    conn_add = backendDb.connect(args.db_file, 'load_to_memory')
    c_add = conn_add.cursor()
    if args.db_rootdir is not None:
        _moveRootDir(c_add, args.db_rootdir, args.rootdir)

    c_add.execute('SELECT COUNT(1) FROM images')
    if (c_add.fetchone()[0] == 0):
        logging.error('Added detabase "%s" has no images.', args.db_file)
        return

    # Copy images.
    c_add.execute('SELECT * FROM images')
    for image_entry_add in c_add.fetchall():
        imagefile = backendDb.imageField(image_entry_add, 'imagefile')

        c.execute('SELECT * FROM images WHERE imagefile=?', (imagefile, ))
        image_entry = c.fetchone()
        if not image_entry:
            logging.debug('Imagefile will be added: %s', imagefile)
            c.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',
                      image_entry_add)
        else:
            logging.debug('Imagefile will be updated: %s', imagefile)
            _maybeUpdateImageField(image_entry, image_entry_add, 'maskfile')
            _maybeUpdateImageField(image_entry, image_entry_add, 'name')
            _maybeUpdateImageField(image_entry, image_entry_add, 'score')
            c.execute(
                'UPDATE images SET width=?, height=?, maskfile=?, '
                'timestamp=?, name=?, score=? WHERE imagefile=?',
                image_entry[1:] + image_entry[0:1])

    # Find the max "match" in "matches" table. New matches will go after that value.
    c.execute('SELECT MAX(match) FROM matches')
    max_match = c.fetchone()[0]
    if max_match is None:
        max_match = 0

    # Copy all other tables.
    c_add.execute('SELECT * FROM objects')
    logging.info('Copying objects.')
    for object_entry_add in progressbar(c_add.fetchall()):
        objectid_add = backendDb.objectField(object_entry_add, 'objectid')

        # Copy objects.
        c.execute(
            'INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
            'VALUES (?,?,?,?,?,?,?)', object_entry_add[1:])
        objectid = c.lastrowid

        # Copy properties.
        c_add.execute('SELECT key,value FROM properties WHERE objectid=?',
                      (objectid_add, ))
        for key, value in c_add.fetchall():
            c.execute(
                'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                (objectid, key, value))

        # Copy polygons.
        c_add.execute('SELECT x,y,name FROM polygons WHERE objectid=?',
                      (objectid_add, ))
        for x, y, name in c_add.fetchall():
            c.execute(
                'INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)',
                (objectid, x, y, name))

        # Copy matches.
        c_add.execute('SELECT match FROM matches WHERE objectid=?',
                      (objectid_add, ))
        for match, in c_add.fetchall():
            c.execute('INSERT INTO matches(objectid,match) VALUES (?,?)',
                      (objectid, match + max_match))


def subtractDbParser(subparsers):
    parser = subparsers.add_parser(
        'subtractDb',
        description='Removes images present in "db_file" from the database. '
        'Objects in the removed images are removed too.')
    parser.set_defaults(func=subtractDb)
    parser.add_argument('--db_file', required=True)
    parser.add_argument(
        '--db_rootdir',
        help=
        'If specified, imagefiles from add_db are considered relative to db_rootdir. '
        'They will be modified to be relative to rootdir of the active database.'
    )


def subtractDb(c, args):

    # Get all the subtracted imagefiles
    c.execute('ATTACH ? AS "subtracted"', (args.db_file, ))
    c.execute('SELECT imagefile from subtracted.images')
    imagefiles_subtracted = c.fetchall()
    c.execute('DETACH DATABASE "subtracted"')

    for imagefile, in progressbar(imagefiles_subtracted):
        # Maybe, change rootdir of imagefile, so that it matches our dataset.
        if args.db_rootdir is not None:
            imagefile = op.relpath(op.join(args.db_rootdir, imagefile),
                                   args.rootdir)
        # If present, delete imagefile and all its objects.
        c.execute('SELECT COUNT(1) FROM images WHERE imagefile=?;',
                  (imagefile, ))
        if c.fetchone()[0] != 0:
            backendDb.deleteImage(c, imagefile)


def splitDbParser(subparsers):
    parser = subparsers.add_parser(
        'splitDb',
        description='Split a db into several sets (randomly or sequentially).')
    parser.set_defaults(func=splitDb)
    parser.add_argument('--out_dir',
                        default='.',
                        help='Common directory for all output databases')
    parser.add_argument('--out_names',
                        required=True,
                        nargs='+',
                        help='Output database names with ".db" extension.')
    parser.add_argument('--out_fractions',
                        required=True,
                        nargs='+',
                        type=float,
                        help='Fractions to put to each output db. '
                        'If sums to >1, last databases will be underfilled.')
    parser.add_argument('--randomly', action='store_true')
    parser.add_argument(
        '--seed',
        type=int,
        help=
        'If specified, used to seed random generator. Can be used with "randomly"'
    )


def splitDb(c, args):
    c.execute('SELECT imagefile FROM images ORDER BY imagefile')
    imagefiles = c.fetchall()
    if args.randomly:
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(imagefiles)

    assert len(args.out_names) == len(args.out_fractions), \
            ('Sizes of not equal: %d != %d' %
            (len(args.out_names), len(args.out_fractions)))

    current = 0
    for db_out_name, fraction in zip(args.out_names, args.out_fractions):
        logging.info((db_out_name, fraction))
        num_images_in_set = int(ceil(len(imagefiles) * fraction))
        next = min(current + num_images_in_set, len(imagefiles))

        # Create a database, and connect to it.
        # Not using ATTACH, because I don't want to commit to the open database.
        logging.info('Writing %d images to %s' %
                     (num_images_in_set, db_out_name))
        db_out_path = op.join(args.out_dir, db_out_name)
        if op.exists(db_out_path):
            os.remove(db_out_path)
        conn_out = sqlite3.connect(db_out_path)
        backendDb.createDb(conn_out)
        c_out = conn_out.cursor()

        for imagefile, in imagefiles[current:next]:

            c.execute('SELECT * FROM images WHERE imagefile=?', (imagefile, ))
            c_out.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?)',
                          c.fetchone())

            c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
            for object_entry in c.fetchall():
                c_out.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?)',
                              object_entry)

            c.execute(
                'SELECT * FROM properties WHERE objectid IN '
                '(SELECT objectid FROM objects WHERE imagefile=?)',
                (imagefile, ))
            for property_entry in c.fetchall():
                c_out.execute('INSERT INTO properties VALUES (?,?,?,?)',
                              property_entry)

            c.execute(
                'SELECT * FROM polygons WHERE objectid IN '
                '(SELECT objectid FROM objects WHERE imagefile=?)',
                (imagefile, ))
            for polygon_entry in c.fetchall():
                c_out.execute('INSERT INTO polygons VALUES (?,?,?,?,?)',
                              polygon_entry)

            c.execute(
                'SELECT * FROM matches WHERE objectid IN '
                '(SELECT objectid FROM objects WHERE imagefile=?)',
                (imagefile, ))
            for match_entry in c.fetchall():
                c_out.execute('INSERT INTO matches VALUES (?,?,?)',
                              match_entry)

        current = next
        conn_out.commit()
        conn_out.close()


def _mergeNObjects(c, objectids):
    ' Merge N objects given by their objectids. '

    objectid_new = objectids[0]
    logging.debug('Merging objects %s into object %d' %
                  (str(objectids), objectid_new))

    # No duplicates.
    if len(objectids) == 1:
        return

    # String from the list.
    objectids_str = (','.join([str(x) for x in objectids]))

    # Merge properties.
    for objectid in objectids[1:]:
        c.execute('UPDATE properties SET objectid=? WHERE objectid=?;',
                  (objectid_new, objectid))
    c.execute(
        'SELECT key,COUNT(DISTINCT(value)) FROM properties WHERE objectid=? GROUP BY key',
        (objectid_new, ))
    for key, num in c.fetchall():
        if num > 1:
            logging.debug(
                'Deleting %d duplicate properties for key=%s and objectid=%s' %
                (num, key, objectid_new))
            c.execute('DELETE FROM properties WHERE objectid=? AND key=?',
                      (objectid_new, key))

    # Merge polygons by adding all the polygons together.
    c.execute(
        'UPDATE polygons SET objectid=?, name=(IFNULL(name, "") || objectid) '
        'WHERE objectid IN (%s)' % objectids_str, (objectid_new, ))

    # Merge matches.
    # Delete matches between only matched objects.
    c.execute(
        'DELETE FROM matches WHERE match NOT IN '
        '(SELECT DISTINCT(match) FROM matches WHERE objectid NOT IN (%s))' %
        objectids_str)
    # Merge matches from all merged to other objects.
    c.execute(
        'UPDATE matches SET objectid=? WHERE objectid IN (%s);' %
        objectids_str, (objectid_new, ))
    # Matches are not necessarily distinct.
    # TODO: Remove duplicate or particially duplicate matches.

    # Merge score.
    c.execute('SELECT DISTINCT(score) FROM objects '
              'WHERE objectid IN (%s) AND score IS NOT NULL' % objectids_str)
    scores = [x for x, in c.fetchall()]
    logging.debug('Merged objects %s have scores %s' % (objectids_str, scores))
    if len(scores) > 1:
        logging.debug(
            'Several distinct score values (%s) for objectids (%s). Will set NULL.'
            % (str(scores), objectids_str))
        c.execute('UPDATE objects SET score=NULL WHERE objectid=?',
                  (objectid_new, ))
    elif len(scores) == 1:
        logging.debug('Writing score=%s to objectid=%s.' %
                      (scores[0], objectid_new))
        c.execute('UPDATE objects SET score=? WHERE objectid=?',
                  (scores[0], objectid_new))
    else:
        logging.debug('No score found for objectid=%s.' % (objectid_new, ))

    # Merge name.
    c.execute(
        'SELECT DISTINCT(name) FROM objects WHERE objectid IN (%s) AND name IS NOT NULL'
        % objectids_str)
    names = [x for x, in c.fetchall()]
    logging.debug('Merged objects %s have names %s' % (objectids_str, names))
    if len(names) > 1:
        logging.debug(
            'Several distinct name values (%s) for objectids (%s). Will set NULL.'
            % (str(names), objectids_str))
        c.execute('UPDATE objects SET name=NULL WHERE objectid=?;',
                  (objectid_new, ))
    elif len(names) == 1:
        logging.debug('Writing name=%s to objectid=%s.' %
                      (names[0], objectid_new))
        c.execute('UPDATE objects SET name=? WHERE objectid=?;',
                  (names[0], objectid_new))
    else:
        logging.debug('No name found for objectid=%s.' % (objectid_new, ))

    # Delete all duplicates from "objects" table. (i.e. except for the 1st object).
    c.execute(
        'DELETE FROM objects WHERE objectid IN (%s) AND objectid != ?' %
        objectids_str, (objectid_new, ))
    return objectid_new


def mergeIntersectingObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'mergeIntersectingObjects',
        description='Merge objects that intersect. '
        'Currently only pairwise, does not merge groups (that is future work.) '
        'Currently implements only intersection by bounding boxes. '
        'A merged object has polygons and properties from both source objects.'
    )
    parser.set_defaults(func=mergeIntersectingObjects)
    parser.add_argument(
        '--IoU_threshold',
        type=float,
        default=0.5,
        help='Intersection over Union threshold to consider merging.')
    parser.add_argument(
        '--target_name',
        help='Name to assign to merged objects. If not specified, '
        'the name is assigned only if the names of the two merged objects match.'
    )
    parser.add_argument(
        '--where_object1',
        default='TRUE',
        help=
        'SQL "where" clause for the "from" objects. The default is any object. '
        'Objects "from" are merged with objects "to".')
    parser.add_argument(
        '--where_object2',
        default='TRUE',
        help=
        'SQL "where" clause for the "to" objects. The default is any object. '
        'Objects "from" are merged with objects "to".')
    parser.add_argument(
        '--where_image',
        default='TRUE',
        help='the SQL "where" clause for "images" table. '
        'E.g. to change imagefile of JPG pictures from directory '
        '"from/mydir" only, use: \'imagefile LIKE "from/mydir/%%"\'')
    parser.add_argument('--with_display',
                        action='store_true',
                        help='Until <Esc> key, display merged opjects.')


def mergeIntersectingObjects(c, args):

    if args.with_display:
        imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # Make polygons if necessary
    logging.info('Syncing bboxes and polygons.')
    polygonsToBboxes(c, None)
    bboxesToPolygons(c, None)

    c.execute('SELECT imagefile FROM images WHERE (%s)' % args.where_image)
    for imagefile, in progressbar(c.fetchall()):
        logging.debug('Processing imagefile %s' % imagefile)

        # Get objects of interest from imagefile.
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_object1, (imagefile, ))
        objects1 = c.fetchall()
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_object2, (imagefile, ))
        objects2 = c.fetchall()
        logging.debug('Image %s has %d and %d objects to match' %
                      (imagefile, len(objects1), len(objects2)))

        pairs_to_merge = util.getIntersectingObjects(objects1,
                                                     objects2,
                                                     args.IoU_threshold,
                                                     same_id_ok=False)

        if args.with_display and len(pairs_to_merge) > 0:
            image = imreader.imread(imagefile)

        # Merge pairs.
        for objectid1, objectid2 in pairs_to_merge:
            # Get rois.
            c.execute('SELECT * WHERE objectid=?', objectid1)
            old_roi1 = backendDb.objectField(c.fetchone(), 'roi')
            c.execute('SELECT * WHERE objectid=?', objectid2)
            old_roi2 = backendDb.objectField(c.fetchone(), 'roi')

            if args.with_display:
                util.drawScoredRoi(image, old_roi1, score=0)
                util.drawScoredRoi(image, old_roi2, score=0.25)

            # Change the name to the target first.
            if args.target_name is not None:
                c.execute('UPDATE objects SET name=? WHERE objectid IN (?,?);',
                          (args.target_name, objectid1, objectid2))

            #logging.info('Merging objects %d and %d.' % (objectid1, objectid2))
            new_objectid = _mergeNObjects(c, [objectid1, objectid2])
            util.polygons2bboxes(c, new_objectid)

            c.execute('SELECT * FROM objects WHERE objectid=?',
                      (new_objectid, ))
            new_roi = backendDb.objectField(c.fetchone(), 'roi')
            logging.info('Merged ROIs %s and %s to make one %s' %
                         (old_roi1, old_roi2, new_roi))

            if args.with_display:
                util.drawScoredRoi(image, new_roi, score=1)

        if args.with_display and len(pairs_to_merge) > 0:
            logging.getLogger().handlers[0].flush()
            cv2.imshow('mergeIntersectingObjects', image[:, :, ::-1])
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyWindow('mergeIntersectingObjects')
                args.with_display = False


def syncObjectidsWithDbParser(subparsers):
    parser = subparsers.add_parser(
        'syncObjectidsWithDb',
        description='Updates objectids with objectids from the reference db: '
        'If an object is found in the reference db (found via IoU), update it. '
        'Otherwise, assign a unique objectid, which is not in either dbs. '
        'Only works on the intersection of imagefiles between dbs.')
    parser.set_defaults(func=syncObjectidsWithDb)
    parser.add_argument(
        '--IoU_threshold',
        type=float,
        default=0.5,
        help='Intersection over Union threshold to consider merging.')
    parser.add_argument('--ref_db_file', required=True)


def syncObjectidsWithDb(c, args):
    def _getNextObjectidInDb(c):
        # Find the maximum objectid in this db. Will create new objectids greater.
        c.execute('SELECT MAX(objectid) FROM objects')
        objectid = c.fetchone()
        # If there are no objects in this db, assign 0.
        return 0 if objectid[0] is None else objectid[0] + 1

    if not op.exists(args.ref_db_file):
        raise FileNotFoundError('Ref db does not exist: %s', args.ref_db_file)
    conn_ref = sqlite3.connect('file:%s?mode=ro' % args.ref_db_file, uri=True)
    c_ref = conn_ref.cursor()

    # Get imagefiles.
    c.execute('SELECT imagefile FROM images')
    imagefiles_this = c.fetchall()
    c_ref.execute('SELECT imagefile FROM images')
    imagefiles_ref = c_ref.fetchall()
    logging.info('The open db has %d, and the ref db has %d imagefiles.',
                 len(imagefiles_this), len(imagefiles_ref))
    # Works only on the intersection of imagefiles.
    imagefiles = list(set(imagefiles_this) & set(imagefiles_ref))
    if not len(imagefiles):
        raise ValueError('No matching imagefiles in between the open db with '
                         '%d images and the reference db with %d images.' %
                         (len(imagefiles_this), len(imagefiles_ref)))
    # Some debug logging.
    imagefiles_this_not_ref = list(set(imagefiles_this) - set(imagefiles_ref))
    if imagefiles_this_not_ref:
        logging.warning(
            'Imagefiles that are in the current db, '
            'but not in the ref db: \n\t%s',
            '\t\n'.join([x for x, in imagefiles_this_not_ref]))
    imagefiles_ref_not_this = list(set(imagefiles_ref) - set(imagefiles_this))
    if imagefiles_ref_not_this:
        logging.warning(
            'Imagefiles that are in the ref db, but not in the '
            'current db: \n\t%s',
            '\t\n'.join([x for x, in imagefiles_ref_not_this]))

    # ASSUME: all objects in the open and the ref db have bboxes values.

    # The new available objectid should be above the max of either db.
    next_objectid = max(_getNextObjectidInDb(c), _getNextObjectidInDb(c_ref))

    # Retire "objects" & "polygons" to avoid objectid conflict during renaming.
    c.execute('ALTER TABLE objects RENAME TO objects_old')
    backendDb.createTableObjects(c)
    c.execute('ALTER TABLE polygons RENAME TO polygons_old')
    backendDb.createTablePolygons(c)

    num_common, num_new = 0, 0
    for imagefile, in progressbar(imagefiles):
        logging.debug('Processing imagefile %s' % imagefile)

        # Get objects of interest from imagefile.
        c.execute('SELECT * FROM objects_old WHERE imagefile=?', (imagefile, ))
        objects = c.fetchall()
        c_ref.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        objects_ref = c_ref.fetchall()
        logging.debug('Image %s has %d and %d objects to match' %
                      (imagefile, len(objects), len(objects_ref)))
        objectid_this_to_ref_map = dict(
            util.getIntersectingObjects(objects, objects_ref,
                                        args.IoU_threshold))
        logging.debug(pformat(objectid_this_to_ref_map))

        for object_ in objects:
            objectid = backendDb.objectField(object_, 'objectid')
            if objectid in objectid_this_to_ref_map:
                new_objectid = objectid_this_to_ref_map[objectid]
                num_common += 1
            else:
                new_objectid = next_objectid
                next_objectid += 1
                num_new += 1
            logging.debug('Inserting id %d in place of %d', new_objectid,
                          objectid)
            object_ = list(object_)
            object_[0] = new_objectid
            object_ = tuple(object_)
            c.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?)', object_)
            c.execute(
                'INSERT INTO polygons(objectid,x,y,name) '
                'SELECT ?,x,y,name FROM polygons_old WHERE objectid=?',
                (new_objectid, objectid))

    logging.info('Found %d common and %d new objects', num_common, num_new)

    c.execute('DROP TABLE objects_old;')
    c.execute('DROP TABLE polygons_old;')

    conn_ref.close()


def syncObjectsDataWithDbParser(subparsers):
    parser = subparsers.add_parser(
        'syncObjectsDataWithDb',
        description='Copy all the data except "imagefile" from "objects" table'
        ' from objects in ref_db_file with matching objectid.')
    parser.set_defaults(func=syncObjectsDataWithDb)
    parser.add_argument('--ref_db_file', required=True)
    parser.add_argument('--cols', nargs='+', help='Columns to update.')


def syncObjectsDataWithDb(c, args):
    if not op.exists(args.ref_db_file):
        raise FileNotFoundError('Ref db does not exist: %s', args.ref_db_file)
    conn_ref = sqlite3.connect('file:%s?mode=ro' % args.ref_db_file, uri=True)
    c_ref = conn_ref.cursor()

    # TODO: Rewrite with joins. Should not need a loop.

    # For logging.
    c.execute('SELECT COUNT(1) FROM objects')
    num_objects = c.fetchone()[0]

    # Get ref objects to copy from.
    c_ref.execute('SELECT * FROM objects ORDER BY objectid')
    objects_ref = c_ref.fetchall()

    count = 0
    count_different_by_col = {col: 0 for col in args.cols}
    for object_ref in objects_ref:
        objectid = backendDb.objectField(object_ref, 'objectid')
        logging.debug('Ref objectid: %d', objectid)
        c.execute('SELECT COUNT(1) FROM objects WHERE objectid=?',
                  (objectid, ))
        does_exist = c.fetchone()[0]
        if not does_exist:
            logging.debug('Object does not exist in the current db.')
            continue
        count += 1
        for col in args.cols:
            value = backendDb.objectField(object_ref, col)
            c.execute('SELECT %s FROM objects WHERE objectid=?' % col,
                      (objectid, ))
            if c.fetchone()[0] != value:
                count_different_by_col[col] += 1
            c.execute('UPDATE objects SET %s=? WHERE objectid=?' % col,
                      (value, objectid))

    # TODO: update polygons, properties, and matches too.

    logging.info('Updated %d objects out of %d. Ref_db had %d more objects.',
                 count, num_objects,
                 len(objects_ref) - count)

    conn_ref.close()
    logging.info(pformat(count_different_by_col))


def renameObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'renameObjects',
        description='Map object names. Can delete names not to be mapped.'
        'Can be used to make an imported dataset compatible with the database.'
    )
    parser.set_defaults(func=renameObjects)
    parser.add_argument(
        '--names_dict',
        required=True,
        help=
        'Map from old names to new names. E.g. \'{"Car": "car", "Truck": "car"}\''
    )
    parser.add_argument('--where_object',
                        default='TRUE',
                        help='SQL "where" clause for the "objects" table.')
    parser.add_argument(
        '--discard_other_objects',
        action='store_true',
        help='Discard objects with names neither in the keys or the values '
        'of "names_dict".')


def renameObjects(c, args):
    namesmap = literal_eval(args.names_dict)

    # Maybe delete other objects.
    if args.discard_other_objects:
        c.execute(
            'SELECT objectid FROM objects WHERE name NOT IN (%s) AND (%s)' %
            (namesmap.keys() + namesmap.values(), args.where_object))
        other_objectids = c.fetchall()
        logging.info('Will delete %d objects with names not in the map.')
        for objectid, in other_objectids:
            backendDb.deleteObject(c, objectid)

    # Remap the rest.
    for key, value in namesmap.items():
        c.execute(
            'UPDATE objects SET name=? WHERE name=? AND (%s)' %
            args.where_object, (value, key))


def resizeAnnotationsParser(subparsers):
    parser = subparsers.add_parser(
        'resizeAnnotations',
        description='Resize all information about images and objects. '
        'Use when images were scaled and the annotations need to be updated. '
        'This command does not scale images themselves. '
        'The "old" image size is obtrained from "images" table.')
    parser.set_defaults(func=resizeAnnotations)
    parser.add_argument(
        '--where_image',
        default='TRUE',
        help='the SQL "where" clause for "images" table. '
        'E.g. to change imagefile of JPG pictures from directory '
        '"from/mydir" only, use: \'imagefile LIKE "from/mydir/%%"\'')
    parser.add_argument('--target_width',
                        type=int,
                        required=False,
                        help='The width each image was scaled to.')
    parser.add_argument('--target_height',
                        type=int,
                        required=False,
                        help='The height each image was scaled to.')


def resizeAnnotations(c, args):
    if args.target_width is None and args.target_height is None:
        raise ValueError(
            'One or both "target_width", "target_height" should be specified.')

    # Process image by image.
    c.execute('SELECT imagefile,width,height FROM images WHERE (%s)' %
              args.where_image)
    for imagefile, old_width, old_height in progressbar(c.fetchall()):
        _resizeImageAnnotations(c, imagefile, old_width, old_height,
                                args.target_width, args.target_height)


def _resizeImageAnnotations(c, imagefile, old_width, old_height, target_width,
                            target_height):

    # Figure out scaling, depending on which of target_width and
    # target_height is given.
    if target_width is not None and target_height is not None:
        percent_x = target_width / float(old_width)
        percent_y = target_height / float(old_height)
        target_width = target_width
        target_height = target_height
    if target_width is None and target_height is not None:
        percent_y = target_height / float(old_height)
        percent_x = percent_y
        target_width = int(old_width * percent_x)
        target_height = target_height
    if target_width is not None and target_height is None:
        percent_x = target_width / float(old_width)
        percent_y = percent_x
        target_width = target_width
        target_height = int(old_height * percent_y)
    logging.debug('Scaling "%s" with percent_x=%.2f, percent_y=%.2f' %
                  (imagefile, percent_x, percent_y))

    # Update image.
    c.execute('UPDATE images SET width=?,height=? WHERE imagefile=?',
              (target_width, target_height, imagefile))

    # Update objects.
    c.execute(
        'SELECT objectid,x1,y1,width,height FROM objects WHERE imagefile=?',
        (imagefile, ))
    for objectid, x1, y1, width, height in c.fetchall():
        if x1 is not None:
            x1 = int(x1 * percent_x)
        if y1 is not None:
            y1 = int(y1 * percent_y)
        if width is not None:
            width = int(width * percent_x)
        if height is not None:
            height = int(height * percent_y)
        c.execute(
            'UPDATE objects SET x1=?,y1=?,width=?,height=? WHERE objectid=?',
            (x1, y1, width, height, objectid))

    # Update polygons.
    c.execute(
        'SELECT id,x,y FROM polygons p INNER JOIN objects o '
        'ON o.objectid=p.objectid WHERE imagefile=?', (imagefile, ))
    for id_, x, y in c.fetchall():
        x = int(x * percent_x)
        y = int(y * percent_y)
        c.execute('UPDATE polygons SET x=?,y=? WHERE id=?', (x, y, id_))


def propertyToNameParser(subparsers):
    parser = subparsers.add_parser('propertyToName',
                                   description='Assigns names to propert')
    parser.set_defaults(func=propertyToName)
    parser.add_argument('--property',
                        required=True,
                        help='Property name to assign to objects.')
    parser.add_argument('--assign_null',
                        action='store_true',
                        help='When specifies, the name is assigned null '
                        'when property is absent for that object.')
    parser.add_argument('--delete_property_after',
                        action='store_true',
                        help='When specifies, deletes that property when done')


def propertyToName(c, args):
    if args.assign_null:
        c.execute(
            'UPDATE objects SET name = ('
            'SELECT value FROM properties '
            'WHERE objects.objectid = properties.objectid AND key=?'
            ')', (args.property, ))
    else:
        c.execute(
            'UPDATE objects SET name = ('
            'SELECT value FROM properties '
            'WHERE objects.objectid = properties.objectid AND key=?'
            ') WHERE objectid IN ('
            'SELECT objectid FROM properties WHERE key=?'
            ')', (args.property, args.property))

    if args.delete_property_after:
        c.execute('DELETE FROM properties WHERE key=?', (args.property, ))


def revertObjectTransformsParser(subparsers):
    parser = subparsers.add_parser(
        'revertObjectTransforms',
        description=
        'Objects (i.e. bboxes and polygons) may have undergone transforms. '
        'These transforms may have been logged into "properties" table. '
        'Tranforms are individual for each object. This subcommand reverts '
        'transforms and restores the original bboxes and polygons. '
        'After that, transforms are cleared from the "properties" table.')
    parser.set_defaults(func=revertObjectTransforms)


def revertObjectTransforms(c, args):
    count = 0
    c.execute('SELECT * FROM objects')
    object_entries = c.fetchall()

    for object_entry in progressbar(object_entries):
        objectid = backendDb.objectField(object_entry, 'objectid')
        # Get the transform.
        c.execute('SELECT value FROM properties WHERE objectid=? AND key="ky"',
                  (objectid, ))
        ky = c.fetchone()
        ky = float(ky[0]) if ky is not None else 1.
        c.execute('SELECT value FROM properties WHERE objectid=? AND key="kx"',
                  (objectid, ))
        kx = c.fetchone()
        kx = float(kx[0]) if kx is not None else 1.
        c.execute('SELECT value FROM properties WHERE objectid=? AND key="by"',
                  (objectid, ))
        by = c.fetchone()
        by = float(by[0]) if by is not None else 0.
        c.execute('SELECT value FROM properties WHERE objectid=? AND key="bx"',
                  (objectid, ))
        bx = c.fetchone()
        bx = float(bx[0]) if bx is not None else 0.
        transform = np.array([[ky, 0., by], [0., kx, bx], [0., 0., 1.]])
        # Get the inverse tranform.
        transform_inv = np.linalg.inv(transform)
        if np.allclose(transform, transform_inv):
            logging.debug('Objectid %d had an identity transform.')
            continue

        count += 1
        # Apply the inverse transform to the bbox and polygon.
        ky = transform_inv[0, 0]
        kx = transform_inv[1, 1]
        by = transform_inv[0, 2]
        bx = transform_inv[1, 2]
        c.execute(
            'UPDATE objects SET x1 = x1 * ? + ?, y1 = y1 * ? + ?, '
            'width = width * ?, height = height * ? WHERE objectid=?',
            (kx, bx, ky, by, kx, ky, objectid))
        c.execute(
            'UPDATE polygons SET x = x * ? + ?, y = y * ? + ? WHERE objectid=?',
            (kx, bx, ky, by, objectid))

        # If imagefile was written to properties, update it too.
        c.execute(
            'SELECT value FROM properties WHERE objectid=? AND key="old_imagefile"',
            (objectid, ))
        old_imagefile = c.fetchone()
        if old_imagefile is not None:
            c.execute('UPDATE objects SET imagefile=? WHERE objectid=?',
                      (old_imagefile[0], objectid))

    c.execute(
        'DELETE FROM properties WHERE key IN ("kx", "ky", "bx", "by", "old_objectid")'
    )
    c.execute('DELETE FROM properties WHERE key="old_imagefile"')
    logging.info('%d objects out of %d were reverted.', count,
                 len(object_entries))
