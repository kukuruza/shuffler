import os.path as op
import numpy as np
import sqlite3
import cv2
import logging
import traceback
import multiprocessing
import concurrent.futures
from progressbar import progressbar

from shuffler.backend import backend_db
from shuffler.backend import backend_media
from shuffler.utils import general as general_utils
from shuffler.utils import boxes as boxes_utils
from shuffler.utils import parser as parser_utils


def add_parsers(subparsers):
    filterImagesViaAnotherDbParser(subparsers)
    filterImagesWithoutObjectsParser(subparsers)
    filterImagesSQLParser(subparsers)
    filterBadImagesParser(subparsers)
    filterObjectsAtImageEdgesParser(subparsers)
    filterObjectsByIntersectionParser(subparsers)
    filterObjectsByNameParser(subparsers)
    filterObjectsByScoreParser(subparsers)
    filterObjectsInsideCertainObjectsParser(subparsers)
    filterObjectsSQLParser(subparsers)


def filterImagesViaAnotherDbParser(subparsers):
    parser = subparsers.add_parser(
        'filterImagesViaAnotherDb',
        description=
        'Remove images from the db that are / are not in the reference db.')
    parser.set_defaults(func=filterImagesViaAnotherDb)
    parser.add_argument(
        '--ref_db_file',
        help=
        'Imagefile entries from this .db file will be kept / deleted from the open db.'
    )
    parser.add_argument(
        '--dirtree_level',
        type=int,
        help='If specified, will use this many levels of the directory '
        'structure for comparison. '
        'E.g. consider imagefiles "my/fancy/image.jpg" and "other/image.jpg". '
        'If dirtree_level=1, they match because filenames "image.jpg" match. '
        'If dirtree_level=2, they do NOT match because "fancy/image.jpg" is '
        'different from "other/image.jpg". '
        'Useful when images in different dirs have the same filename. '
        'If not specified (default), will compare the whole path.')
    parser_utils.addKeepOrDeleteArguments(parser)


def takeSubpath(path, dirtree_level=None):
    ' Takes dirtree_level parts from the path from the end and removes os.sep. '
    if dirtree_level is None:
        return path
    path = path[::-1]
    path = path.replace(op.sep, "", dirtree_level - 1)
    path = path[::-1]
    return op.basename(path)


def filterImagesViaAnotherDb(c, args):
    # Get all the imagefiles from the reference db.
    if not op.exists(args.ref_file):
        raise FileNotFoundError('Reference db does not exist: %s' %
                                args.ref_file)
    conn_ref = sqlite3.connect('file:%s?mode=ro' % args.ref_file, uri=True)
    c_ref = conn_ref.cursor()
    c_ref.execute('SELECT imagefile FROM images')
    imagefiles_ref = [imagefile for imagefile, in c_ref.fetchall()]
    logging.info('Total %d images in ref.', len(imagefiles_ref))
    conn_ref.close()

    # Get all the imagefiles from the main db.
    c.execute('SELECT imagefile FROM images')
    imagefiles = [imagefile for imagefile, in c.fetchall()]
    logging.info('Before filtering have %d files', len(imagefiles))

    # imagefiles_del are either must be or must not be in the other database.
    if args.keep:
        imagefiles_del = [
            x for x in imagefiles
            if takeSubpath(x, args.dirtree_level) not in imagefiles_ref
        ]
    else:
        imagefiles_del = [
            x for x in imagefiles
            if takeSubpath(x, args.dirtree_level) in imagefiles_ref
        ]
    logging.info('Will delete %d images', len(imagefiles_del))

    # Delete.
    for imagefile_del in imagefiles_del:
        backend_db.deleteImage(c, imagefile_del)

    # Get all the imagefiles from the main db.
    c.execute('SELECT COUNT(*) FROM images')
    count = c.fetchone()[0]
    logging.info('%d images left.', count)


def filterImagesWithoutObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'filterImagesWithoutObjects',
        description='Delete images that have no objects.')
    parser.set_defaults(func=filterImagesWithoutObjects)
    parser_utils.addWhereImageArgument(parser)


def filterImagesWithoutObjects(c, args):
    c.execute('SELECT imagefile FROM images WHERE (%s) AND imagefile NOT IN '
              '(SELECT imagefile FROM objects)' % args.where_image)
    for imagefile, in progressbar(c.fetchall()):
        backend_db.deleteImage(c, imagefile)


def filterImagesSQLParser(subparsers):
    parser = subparsers.add_parser(
        'filterImagesSQL',
        description=
        'Delete images (and their objects) based on the SQL "where_image" clause.'
    )
    parser.set_defaults(func=filterImagesSQL)
    parser.add_argument(
        '--sql', help='an arbitrary SQL clause that should query "imagefile"')


def filterImagesSQL(c, args):
    c.execute('SELECT COUNT(1) FROM images')
    logging.info('Before filtering have %d images.', c.fetchone()[0])

    c.execute(args.sql)

    imagefiles = c.fetchall()
    logging.info('Will filter %d images.', len(imagefiles))
    for imagefile, in progressbar(imagefiles):
        backend_db.deleteImage(c, imagefile)


def filterBadImagesParser(subparsers):
    parser = subparsers.add_parser(
        'filterBadImages',
        description='Loads all images and masks and delete unreadable ones. '
        'Imagefile records with missing images or masks are not deleted')
    parser.add_argument('--force_single_thread',
                        action='store_true',
                        help='If specified, single thread is used.')
    parser.set_defaults(func=filterBadImages)


def filterBadImages(c, args):
    def isImageOk(imreader, imagefile, maskfile):
        if imagefile is not None:
            try:
                imreader.imread(imagefile)
            except Exception:
                logging.info('Found a bad image %s', imagefile)
                logging.debug("Got exception:\n%s", traceback.print_exc())
                return False
        if maskfile is not None:
            try:
                imreader.maskread(maskfile)
            except Exception:
                logging.info('Found a bad mask %s for imagefile %s', maskfile,
                             imagefile)
                logging.debug("Got exception:\n%s", traceback.print_exc())
                return False
        return True

    imreader = backend_media.MediaReader(rootdir=args.rootdir)

    num_deleted_imagefiles = 0

    c.execute('SELECT imagefile,maskfile FROM images')
    image_entries = c.fetchall()

    if (isinstance(imreader, backend_media.PictureReader)
            and not args.force_single_thread):
        # Can do it parallel.
        max_workers = multiprocessing.cpu_count() - 1
        logging.info('Running filtering with %d workers.', max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            futures = []
            for imagefile, maskfile in progressbar(image_entries):
                logging.debug('Imagefile "%s"', imagefile)
                futures.append(
                    executor.submit(isImageOk, imreader, imagefile, maskfile))

        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                backend_db.deleteImage(c, imagefile)
                num_deleted_imagefiles += 1

    else:
        logging.info('Running filtering in a single thread.')
        for imagefile, maskfile in progressbar(image_entries):
            logging.debug('Imagefile "%s"', imagefile)
            if not isImageOk(imreader, imagefile, maskfile):
                backend_db.deleteImage(c, imagefile)
                num_deleted_imagefiles += 1

    logging.info("Deleted %d image(s).", num_deleted_imagefiles)


# TODO: keep or delete.
def filterObjectsAtImageEdgesParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsAtImageEdges',
        description='Delete objects closer than a threshold from image edges.')
    parser.set_defaults(func=filterObjectsAtImageEdges)
    parser.add_argument(
        '--threshold',
        default=0.03,
        help='Width of the buffer around the image edges specified as a '
        'fraction of image dimensions. Objects intersecting with this buffer '
        'and outside of the image are deleted.')
    parser.add_argument(
        '--display',
        action='store_true',
        help=
        'Until <Esc> key, images are displayed with objects on Border as red, '
        'and other as blue.')


def filterObjectsAtImageEdges(c, args):
    def isPolygonAtImageEdge(polygon_entries, imwidth, imheight, threshold):
        '''
        A polygon is considered to be at image edge iff at least one point is
        within the buffer from the edge.
        '''
        assert len(polygon_entries) > 0
        xs = [backend_db.polygonField(p, 'x') for p in polygon_entries]
        ys = [backend_db.polygonField(p, 'y') for p in polygon_entries]
        x_buffer_depth = imwidth * threshold
        y_buffer_depth = imheight * threshold
        num_too_close = 0
        num_too_close += sum([x < x_buffer_depth for x in xs])
        num_too_close += sum([y < y_buffer_depth for y in ys])
        num_too_close += sum([x > imwidth - x_buffer_depth for x in xs])
        num_too_close += sum([y > imheight - y_buffer_depth for y in ys])
        return num_too_close >= 1

    def isRoiAtImageEdge(roi, imwidth, imheight, threshold):
        '''
        A roi is considered to be at image edge iff one of the sides is within
        a buffer from the edge.
        '''
        x_buffer_depth = imwidth * threshold
        y_buffer_depth = imheight * threshold
        logging.debug('x_buffer_depth: %f, y_buffer_depth: %f', x_buffer_depth,
                      y_buffer_depth)
        assert len(roi) == 4
        return (roi[0] < y_buffer_depth or roi[1] < x_buffer_depth
                or roi[2] > imheight + 1 - y_buffer_depth
                or roi[3] > imwidth + 1 - x_buffer_depth)

    if args.display:
        imreader = backend_media.MediaReader(rootdir=args.rootdir)

    # For the reference.
    c.execute('SELECT COUNT(1) FROM objects')
    num_before = c.fetchone()[0]

    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        if args.display:
            image = imreader.imread(imagefile)

        c.execute('SELECT width,height FROM images WHERE imagefile=?',
                  (imagefile, ))
        (imwidth, imheight) = c.fetchone()

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        object_entries = c.fetchall()
        logging.debug('%d objects found for %s', len(object_entries),
                      imagefile)

        for object_entry in object_entries:
            for_deletion = False

            # Get all necessary entries.
            objectid = backend_db.objectField(object_entry, 'objectid')
            roi = backend_db.objectField(object_entry, 'roi')
            c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid, ))
            polygon_entries = c.fetchall()

            # Find if the polygon or the roi is at the border,
            # polygon has preference over roi.
            if len(polygon_entries) > 0:
                if isPolygonAtImageEdge(polygon_entries, imwidth, imheight,
                                        args.threshold):
                    logging.debug('Found a polygon at edge %s', str(polygon))
                    for_deletion = True
            elif roi is not None:
                if isRoiAtImageEdge(roi, imwidth, imheight, args.threshold):
                    logging.debug('Found a roi at edge %s', str(roi))
                    for_deletion = True
            else:
                logging.error(
                    'Neither polygon, nor bbox is available for objectid %d',
                    objectid)

            # Draw polygon or roi.
            if args.display:
                if len(polygon_entries) > 0:
                    polygon = [(backend_db.polygonField(p, 'x'),
                                backend_db.polygonField(p, 'y'))
                               for p in polygon_entries]
                    general_utils.drawScoredPolygon(
                        image, polygon, score=(0 if for_deletion else 1))
                elif roi is not None:
                    general_utils.drawScoredRoi(
                        image, roi, score=(0 if for_deletion else 1))

            # Delete if necessary
            if for_deletion:
                backend_db.deleteObject(c, objectid)

        if args.display:
            cv2.imshow('filterObjectsAtImageEdges', image[:, :, ::-1])
            key = cv2.waitKey(-1)
            if key == 27:
                args.display = False
                cv2.destroyWindow('filterObjectsAtImageEdges')

    # For the reference.
    c.execute('SELECT COUNT(1) FROM objects')
    num_after = c.fetchone()[0]
    logging.info('Deleted %d out of %d objects.', num_before - num_after,
                 num_before)


# TODO: Should be used for deleting duplicates. Keep the highest score object.
#       Should use hierarchical clustering.
# TODO: Add intersection by polygons.
def filterObjectsByIntersectionParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByIntersection',
        description='Remove objects that intersect with other objects.')
    parser.set_defaults(func=filterObjectsByIntersection)
    parser.add_argument(
        '--IoA_threshold',
        default=0.1,
        type=float,
        help='Minimal Intersection over AREA of the object. If an object '
        'has IoA > IoA_threshold with some other object, it is deleted.')
    parser.add_argument(
        '--display',
        action='store_true',
        help='Until <Esc> key, display how each object intersects others. '
        'Bad objects are shown as as red, good as blue, '
        'while the others as green.')


def filterObjectsByIntersection(c, args):
    def getRoiIntersection(roi1, roi2):
        dy = min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])
        dx = min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])
        if dy <= 0 or dx <= 0:
            return 0
        return dy * dx

    if args.display:
        imreader = backend_media.MediaReader(rootdir=args.rootdir)

    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        if args.display:
            image = imreader.imread(imagefile)

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        object_entries = c.fetchall()
        logging.debug('%d objects found for %s', len(object_entries),
                      imagefile)

        good_objects = np.ones(shape=len(object_entries), dtype=bool)
        for iobject1, object_entry1 in enumerate(object_entries):

            roi1 = backend_db.objectField(object_entry1, 'roi')
            if roi1 is None:
                logging.error(
                    'No roi for objectid %d, intersection on polygons '
                    'not implemented.', iobject1)
                continue

            area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
            if area1 == 0:
                logging.warning('An object in %s has area 0. Will delete.',
                                imagefile)
                good_objects[iobject1] = False
                break

            for iobject2, object_entry2 in enumerate(object_entries):
                if iobject2 == iobject1:
                    continue
                roi2 = backend_db.objectField(object_entry2, 'roi')
                if roi2 is None:
                    continue
                IoA = getRoiIntersection(roi1, roi2) / float(area1)
                if IoA > args.IoA_threshold:
                    good_objects[iobject1] = False
                    break

            if args.display:
                image = imreader.imread(imagefile)
                general_utils.drawScoredRoi(
                    image, roi1, score=(1 if good_objects[iobject1] else 0))
                for iobject2, object_entry2 in enumerate(object_entries):
                    if iobject1 == iobject2:
                        continue
                    roi2 = backend_db.objectField(object_entry2, 'roi')
                    if roi2 is None:
                        continue
                    general_utils.drawScoredRoi(image, roi2, score=0.5)
                cv2.imshow('filterObjectsByIntersection', image[:, :, ::-1])
                key = cv2.waitKey(-1)
                if key == 27:
                    cv2.destroyWindow('filterObjectsByIntersection')
                    args.display = 0

        for object_entry, is_object_good in zip(object_entries, good_objects):
            if not is_object_good:
                backend_db.deleteObject(
                    c, backend_db.objectField(object_entry, 'objectid'))


def filterObjectsByNameParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByName',
        description=
        'Delete objects with specific names or all but with specific names.')
    parser.set_defaults(func=filterObjectsByName)
    parser.add_argument('--names',
                        nargs='+',
                        help='A list of object names to keep or delete.')
    parser_utils.addKeepOrDeleteArguments(parser)


def filterObjectsByName(c, args):
    if args.keep:
        keep_names = ','.join(['"%s"' % x for x in args.names])
        logging.info('Will keep names: %s', keep_names)
        c.execute('SELECT objectid FROM objects WHERE name NOT IN (%s)',
                  keep_names)
    else:
        delete_names = ','.join(['"%s"' % x for x in args.names])
        logging.info('Will delete names: %s', delete_names)
        c.execute('SELECT objectid FROM objects WHERE name IN (%s)',
                  delete_names)
    for objectid, in progressbar(c.fetchall()):
        backend_db.deleteObject(c, objectid)


def filterObjectsByScoreParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByScore',
        description='Delete all objects that have score less than "threshold".'
    )
    parser.set_defaults(func=filterObjectsByScore)
    parser.add_argument('--threshold', type=float, default=0.5)


def filterObjectsByScore(c, args):
    c.execute('SELECT objectid FROM objects WHERE score < %f' % args.threshold)
    for objectid, in progressbar(c.fetchall()):
        backend_db.deleteObject(c, objectid)


# TODO. Should not use center.
def filterObjectsInsideCertainObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsInsideCertainObjects',
        description='Delete objects with CENTERS inside certain other objects.'
    )
    parser.set_defaults(func=filterObjectsInsideCertainObjects)
    parser.add_argument(
        '--where_shadowing_objects',
        required=True,
        help='SQL "where" clause that queries for "objectid". '
        'Everything with the center inside these objects '
        '(subject to "where_object") will be deleted. '
        'Whether a point is "inside" an object is determined by its polygon '
        'if it exists, otherwise the bounding box. '
        'Queries table "objects". Example: \'objects.name == "bus"\'')
    parser_utils.addWhereObjectArgument(parser)
    parser_utils.addKeepOrDeleteArguments(parser)


def filterObjectsInsideCertainObjects(c, args):
    c.execute('SELECT COUNT(1) FROM objects WHERE (%s)' % args.where_object)
    count_before = c.fetchone()[0]
    count_deleted = 0

    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        # Shadow objects.
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_shadowing_objects, (imagefile, ))
        shadow_object_entries = c.fetchall()
        logging.debug('Found %d shadowing objects in imagefile %s',
                      len(shadow_object_entries), imagefile)
        # Populate polygons of the shadow objects.
        shadow_object_polygons = []
        for shadow_object_entry in shadow_object_entries:
            shadow_objectid = backend_db.objectField(shadow_object_entry,
                                                     'objectid')
            c.execute('SELECT y,x FROM polygons WHERE objectid=?',
                      (shadow_objectid, ))
            shadow_polygon = c.fetchall()
            shadow_object_polygons.append(shadow_polygon)
        shadow_object_ids_set = set([
            backend_db.objectField(entry, 'objectid')
            for entry in shadow_object_entries
        ])

        # Get all the objects that can be considered.
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_object, (imagefile, ))
        object_entries = c.fetchall()
        logging.debug('Total %d objects satisfying the condition.',
                      len(object_entries))

        for object_entry in object_entries:
            objectid = backend_db.objectField(object_entry, 'objectid')
            if objectid in shadow_object_ids_set:
                logging.debug('Object %d is in the shadow set', objectid)
                continue
            c.execute('SELECT AVG(y),AVG(x) FROM polygons WHERE objectid=?',
                      (objectid, ))
            center_yx = c.fetchone()
            # If polygon does not exist, use bbox.
            if center_yx[0] is None:
                roi = boxes_utils.bbox2roi(
                    backend_db.objectField(object_entry, 'bbox'))
                center_yx = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
            logging.debug('center_yx: %s', str(center_yx))

            is_inside_any = False
            for shadow_object_entry, shadow_polygon in zip(
                    shadow_object_entries, shadow_object_polygons):
                # Get the shadow roi, or polygon if it exists.
                shadow_objectid = backend_db.objectField(
                    object_entry, 'objectid')

                # Check that the center is within the shadow polygon or bbox.
                if len(shadow_polygon) > 0:
                    is_inside = cv2.pointPolygonTest(
                        np.array(shadow_polygon).astype(int), center_yx,
                        False) >= 0
                    logging.debug('Object %d is %sinside polygon', objectid,
                                  ' ' if is_inside else 'not ')
                else:
                    shadow_roi = boxes_utils.bbox2roi(
                        backend_db.objectField(shadow_object_entry, 'bbox'))
                    is_inside = (center_yx[0] > shadow_roi[0]
                                 and center_yx[0] < shadow_roi[2]
                                 and center_yx[1] > shadow_roi[1]
                                 and center_yx[1] < shadow_roi[3])
                    logging.debug('Object %d is %sinside bbox', objectid,
                                  ' ' if is_inside else 'not ')

                if is_inside:
                    is_inside_any = True
                    break  # One is enough.

            if is_inside_any != args.keep:
                backend_db.deleteObject(c, objectid)
                count_deleted += 1

    logging.info('Deleted %d objects out of %d.', count_deleted, count_before)


def filterObjectsSQLParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsSQL',
        description='Delete objects based on the SQL "where" clause.')
    parser.set_defaults(func=filterObjectsSQL)
    parser.add_argument(
        '--sql',
        help='an arbitrary SQL clause that should query "objectid". E.g.'
        '"SELECT objects.objectid FROM objects '
        'INNER JOIN properties ON objects.objectid=properties.objectid '
        'WHERE properties.value=\'blue\' AND objects.score > 0.8"')


def filterObjectsSQL(c, args):
    c.execute('SELECT COUNT(1) FROM objects')
    logging.info('Before filtering have %d objects.', c.fetchone()[0])

    c.execute(args.sql)

    objectids = c.fetchall()
    logging.info('Going to remove %d objects.', len(objectids))
    for objectid, in progressbar(objectids):
        backend_db.deleteObject(c, objectid)
