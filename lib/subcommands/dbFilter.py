import os, sys, os.path as op
import numpy as np
import sqlite3
import cv2
import logging
from glob import glob
from pprint import pformat
from progressbar import progressbar

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util
from lib.utils import utilBoxes


def add_parsers(subparsers):
    filterImagesOfAnotherDbParser(subparsers)
    filterObjectsAtBorderParser(subparsers)
    filterObjectsByIntersectionParser(subparsers)
    filterObjectsByNameParser(subparsers)
    filterEmptyImagesParser(subparsers)
    filterObjectsByScoreParser(subparsers)
    filterObjectsInsideCertainObjectsParser(subparsers)
    filterObjectsSQLParser(subparsers)
    filterImagesSQLParser(subparsers)
    filterImagesByIdsParser(subparsers)


def filterImagesOfAnotherDbParser(subparsers):
    parser = subparsers.add_parser(
        'filterImagesOfAnotherDb',
        description=
        'Remove images from the db that are / are not in the reference db.')
    db_group = parser.add_mutually_exclusive_group()
    db_group.add_argument(
        '--keep_db_file',
        help='If specified, will KEEP all images that are in keep_db_file.')
    db_group.add_argument(
        '--delete_db_file',
        help='If specified, will DELETE all images that are in keep_db_file.')
    parser.add_argument(
        '--use_basename',
        action='store_true',
        help='If specified, compare files based on their basename, not paths.')
    parser.set_defaults(func=filterImagesOfAnotherDb)


def filterImagesOfAnotherDb(c, args):
    if args.keep_db_file is None and args.delete_db_file is None:
        raise ValueError(
            'Either "keep_db_file" or "delete_db_file" must be specified.')

    # Get all the imagefiles from the reference db.
    ref_file = args.keep_db_file if args.keep_db_file else args.delete_db_file
    conn_ref = sqlite3.connect('file:%s?mode=ro' % ref_file, uri=True)
    c_ref = conn_ref.cursor()
    c_ref.execute('SELECT imagefile FROM images')
    imagefiles_ref = c_ref.fetchall()
    logging.info('Total %d images in ref.' % len(imagefiles_ref))
    conn_ref.close()

    # Get all the imagefiles from the main db.
    c.execute('SELECT imagefile FROM images')
    imagefiles = c.fetchall()
    logging.info('Before filtering have %d files', len(imagefiles))

    maybe_basename = lambda x: op.basename(x) if args.use_basename else x
    if args.use_basename:
        imagefiles_ref = [op.basename(x) for x, in imagefiles_ref]

    # imagefiles_del are either must be or must not be in the other database.
    if args.keep_db_file is not None:
        imagefiles_del = [
            x for x, in imagefiles if maybe_basename(x) not in imagefiles_ref
        ]
    elif args.delete_db_file is not None:
        imagefiles_del = [
            x for x, in imagefiles if maybe_basename(x) in imagefiles_ref
        ]
    else:
        assert 0, "We cant be here."

    # Delete.
    for imagefile_del in imagefiles_del:
        backendDb.deleteImage(c, imagefile_del)

    # Get all the imagefiles from the main db.
    c.execute('SELECT COUNT(*) FROM images')
    count = c.fetchone()[0]
    logging.info('%d images left.' % count)


def filterObjectsAtBorderParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsAtBorder',
        description='Delete bboxes closer than "border_thresh" from border.')
    parser.set_defaults(func=filterObjectsAtBorder)
    parser.add_argument(
        '--border_thresh',
        default=0.03,
        help='Width of the buffer around the image edges '
        'specified as a percentage of image dimensions'
        'Objects in this buffer and outside of the image are filtered out.')
    parser.add_argument(
        '--with_display',
        action='store_true',
        help=
        'Until <Esc> key, images are displayed with objects on Border as red, '
        'and other as blue.')


def filterObjectsAtBorder(c, args):
    def isPolygonAtBorder(polygon_entries, width, height, border_thresh_perc):
        xs = [backendDb.polygonField(p, 'x') for p in polygon_entries]
        ys = [backendDb.polygonField(p, 'y') for p in polygon_entries]
        border_thresh = (height + width) / 2 * border_thresh_perc
        dist_to_border = min(xs, [width - x for x in xs], ys,
                             [height - y for y in ys])
        num_too_close = sum([x < border_thresh for x in dist_to_border])
        return num_too_close >= 2

    def isRoiAtBorder(roi, width, height, border_thresh_perc):
        border_thresh = (height + width) / 2 * border_thresh_perc
        logging.debug('border_thresh: %f' % border_thresh)
        return min(roi[0], roi[1], height + 1 - roi[2],
                   width + 1 - roi[3]) < border_thresh

    if args.with_display:
        imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For the reference.
    c.execute('SELECT COUNT(1) FROM objects')
    num_before = c.fetchone()[0]

    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        if args.with_display:
            image = imreader.imread(imagefile)

        c.execute('SELECT width,height FROM images WHERE imagefile=?',
                  (imagefile, ))
        (imwidth, imheight) = c.fetchone()

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        object_entries = c.fetchall()
        logging.debug('%d objects found for %s' %
                      (len(object_entries), imagefile))

        for object_entry in object_entries:
            for_deletion = False

            # Get all necessary entries.
            objectid = backendDb.objectField(object_entry, 'objectid')
            roi = backendDb.objectField(object_entry, 'roi')
            c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid, ))
            polygon_entries = c.fetchall()

            # Find if the polygon or the roi is at the border,
            # polygon has preference over roi.
            if len(polygon_entries) > 0:
                if isPolygonAtBorder(polygon_entries, imwidth, imheight,
                                     args.border_thresh):
                    logging.debug('border polygon %s' % str(polygon))
                    for_deletion = True
            elif roi is not None:
                if isRoiAtBorder(roi, imwidth, imheight, args.border_thresh):
                    logging.debug('border roi %s' % str(roi))
                    for_deletion = True
            else:
                logging.error(
                    'Neither polygon, nor bbox is available for objectid %d' %
                    objectid)

            # Draw polygon or roi.
            if args.with_display:
                if len(polygon_entries) > 0:
                    polygon = [(backendDb.polygonField(p, 'x'),
                                backendDb.polygonField(p, 'y'))
                               for p in polygon_entries]
                    util.drawScoredPolygon(image,
                                           polygon,
                                           score=(0 if for_deletion else 1))
                elif roi is not None:
                    util.drawScoredRoi(image,
                                       roi,
                                       score=(0 if for_deletion else 1))

            # Delete if necessary
            if for_deletion:
                backendDb.deleteObject(c, objectid)

        if args.with_display:
            cv2.imshow('filterObjectsAtBorder', image[:, :, ::-1])
            key = cv2.waitKey(-1)
            if key == 27:
                args.with_display = False
                cv2.destroyWindow('filterObjectsAtBorder')

    # For the reference.
    c.execute('SELECT COUNT(1) FROM objects')
    num_after = c.fetchone()[0]
    logging.info('Deleted %d out of %d objects.' %
                 (num_before - num_after, num_before))


def filterObjectsByIntersectionParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByIntersection',
        description='Remove cars that have high intersection with other cars.')
    parser.set_defaults(func=filterObjectsByIntersection)
    parser.add_argument(
        '--intersection_thresh',
        default=0.1,
        type=float,
        help=
        'How much an object has to intersect with others to be filtered out.')
    parser.add_argument(
        '--with_display',
        action='store_true',
        help='Until <Esc> key, display how each object intersects others. '
        'Bad objects are shown as as red, good as blue, '
        'while the others as green.')


def filterObjectsByIntersection(c, args):
    def getRoiIntesection(rioi1, roi2):
        dy = min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])
        dx = min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])
        if dy <= 0 or dx <= 0: return 0
        return dy * dx

    if args.with_display:
        imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        if args.with_display:
            image = imreader.imread(imagefile)

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        object_entries = c.fetchall()
        logging.debug('%d objects found for %s' %
                      (len(object_entries), imagefile))

        good_objects = np.ones(shape=len(object_entries), dtype=bool)
        for iobject1, object_entry1 in enumerate(object_entries):

            #roi1 = _expandCarBbox_(object_entry1, args)
            roi1 = backendDb.objectField(object_entry1, 'roi')
            if roi1 is None:
                logging.error(
                    'No roi for objectid %d, intersection on polygons '
                    'not implemented.', iobject1)
                continue

            area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
            if area1 == 0:
                logging.warning('An object in %s has area 0. Will delete.' %
                                imagefile)
                good_objects[iobject1] = False
                break

            for iobject2, object_entry2 in enumerate(object_entries):
                if iobject2 == iobject1:
                    continue
                roi2 = backendDb.objectField(object_entry2, 'roi')
                if roi2 is None:
                    continue
                intersection = getRoiIntesection(roi1, roi2) / float(area1)
                if intersection > args.intersection_thresh:
                    good_objects[iobject1] = False
                    break

            if args.with_display:
                image = imreader.imread(imagefile)
                util.drawScoredRoi(image,
                                   roi1,
                                   score=(1 if good_objects[iobject1] else 0))
                for iobject2, object_entry2 in enumerate(object_entries):
                    if iobject1 == iobject2:
                        continue
                    roi2 = backendDb.objectField(object_entry2, 'roi')
                    if roi2 is None:
                        continue
                    util.drawScoredRoi(image, roi2, score=0.5)
                cv2.imshow('filterObjectsByIntersection', image[:, :, ::-1])
                key = cv2.waitKey(-1)
                if key == 27:
                    cv2.destroyWindow('filterObjectsByIntersection')
                    args.with_display = 0

        for object_entry, is_object_good in zip(object_entries, good_objects):
            if not is_object_good:
                backendDb.deleteObject(
                    c, backendDb.objectField(object_entry, 'objectid'))


def filterObjectsByNameParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByName',
        description='Filter away car entries with unknown names.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--good_names',
        nargs='+',
        help='A list of object names to keep. Others will be filtered.')
    group.add_argument('--bad_names',
                       nargs='+',
                       help='A list of object names to delete.')
    parser.set_defaults(func=filterObjectsByName)


def filterObjectsByName(c, args):
    if args.good_names is not None:
        good_names = ','.join(['"%s"' % x for x in args.good_names])
        logging.info('Will keep the following object names: %s' % good_names)
        c.execute('SELECT objectid FROM objects WHERE name NOT IN (%s)' %
                  good_names)
    elif args.bad_names is not None:
        bad_names = ','.join(['"%s"' % x for x in args.bad_names])
        logging.info('Will delete the following object names: %s' % bad_names)
        c.execute('SELECT objectid FROM objects WHERE name IN (%s)' %
                  bad_names)
    else:
        raise ValueError('"good_names" or "bad_names" must be specified.')
    for objectid, in progressbar(c.fetchall()):
        backendDb.deleteObject(c, objectid)


def filterEmptyImagesParser(subparsers):
    parser = subparsers.add_parser('filterEmptyImages')
    parser.add_argument(
        '--where_image',
        default='TRUE',
        help='the SQL "where" clause for "images" table. '
        'E.g. to change imagefile of JPG pictures from directory "from/mydir" '
        'only, use: \'imagefile LIKE "from/mydir/%%"\'')
    parser.set_defaults(func=filterEmptyImages)


def filterEmptyImages(c, args):
    c.execute('SELECT imagefile FROM images WHERE (%s) AND imagefile NOT IN '
              '(SELECT imagefile FROM objects)' % args.where_image)
    for imagefile, in progressbar(c.fetchall()):
        backendDb.deleteImage(c, imagefile)


def filterObjectsByScoreParser(subparsers):
    parser = subparsers.add_parser(
        'filterObjectsByScore',
        description=
        'Delete all objects that have score less than "score_threshold".')
    parser.set_defaults(func=filterObjectsByScore)
    parser.add_argument('--score_threshold', type=float, default=0.5)


def filterObjectsByScore(c, args):
    c.execute('SELECT objectid FROM objects WHERE score < %f' %
              args.score_threshold)
    for objectid, in progressbar(c.fetchall()):
        backendDb.deleteObject(c, objectid)


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
        '(subject to "where_objects") will be deleted. '
        'Whether a point is "inside" an object is determined by its polygon '
        'if it exists, otherwise the bounding box. '
        'Queries table "objects". Example: \'objects.name == "bus"\'')
    parser.add_argument(
        '--where_objects',
        default='TRUE',
        help='SQL "where" clause that queries for "objectid" to crop. '
        'Queries table "objects". Example: \'objects.name == "car"\'')


def filterObjectsInsideCertainObjects(c, args):
    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        # Shadow objects.
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_shadowing_objects, (imagefile, ))
        shadow_object_entries = c.fetchall()
        logging.info('Found %d shadowing objects.', len(shadow_object_entries))
        # Populate polygons of the shadow objects.
        shadow_object_polygons = []
        for shadow_object_entry in shadow_object_entries:
            shadow_objectid = backendDb.objectField(shadow_object_entry,
                                                    'objectid')
            c.execute('SELECT y,x FROM polygons WHERE objectid=?',
                      (shadow_objectid, ))
            shadow_polygon = c.fetchall()
            shadow_object_polygons.append(shadow_polygon)
        shadow_object_ids_set = set([
            backendDb.objectField(entry, 'objectid')
            for entry in shadow_object_entries
        ])

        # Get all the objects that can be considered.
        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_objects, (imagefile, ))
        object_entries = c.fetchall()
        logging.info('Total %d objects satisfying the condition.',
                     len(object_entries))

        for object_entry in object_entries:
            objectid = backendDb.objectField(object_entry, 'objectid')
            if objectid in shadow_object_ids_set:
                logging.debug('Object %d is in the shadow set', objectid)
                continue
            c.execute('SELECT AVG(y),AVG(x) FROM polygons WHERE objectid=?',
                      (objectid, ))
            center_yx = c.fetchone()
            # If polygon does not exist, use bbox.
            if center_yx[0] is None:
                roi = utilBoxes.bbox2roi(
                    backendDb.objectField(object_entry, 'bbox'))
                center_yx = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
            logging.debug('center_yx: %s', str(center_yx))

            for shadow_object_entry, shadow_polygon in zip(
                    shadow_object_entries, shadow_object_polygons):
                # Get the shadow roi, or polygon if it exists.
                shadow_objectid = backendDb.objectField(
                    object_entry, 'objectid')

                # Check that the center is within the shadow polygon or bbox.
                if len(shadow_polygon) > 0:
                    is_inside = cv2.pointPolygonTest(np.array(shadow_polygon),
                                                     center_yx, False) >= 0
                else:
                    shadow_roi = utilBoxes.bbox2roi(
                        backendDb.objectField(shadow_object_entry, 'bbox'))
                    is_inside = (center_yx[0] > shadow_roi[0]
                                 and center_yx[0] < shadow_roi[2]
                                 and center_yx[1] > shadow_roi[1]
                                 and center_yx[1] < shadow_roi[3])

                if is_inside:
                    backendDb.deleteObject(c, objectid)
                    # We do not need to check other shadow_object_entries.
                    continue


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
    logging.info('Before filtering have %d objects.' % c.fetchone()[0])

    c.execute(args.sql)

    objectids = c.fetchall()
    logging.info('Going to remove %d objects.' % len(objectids))
    for objectid, in progressbar(objectids):
        backendDb.deleteObject(c, objectid)


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
    logging.info('Before filtering have %d images.' % c.fetchone()[0])

    c.execute(args.sql)

    imagefiles = c.fetchall()
    logging.info('Will filter %d images.', len(imagefiles))
    for imagefile, in progressbar(imagefiles):
        backendDb.deleteImage(c, imagefile)


def filterImagesByIdsParser(subparsers):
    parser = subparsers.add_parser(
        'filterImagesByIds',
        description='Delete images (and their objects) based on their '
        'sequential number in the sorted table.')
    parser.set_defaults(func=filterImagesByIds)
    parser.add_argument('ids',
                        type=int,
                        nargs='+',
                        help='A sequential number of an image in the database')


def filterImagesByIds(c, args):
    c.execute('SELECT imagefile FROM images ORDER BY imagefile')
    imagefiles = c.fetchall()
    imagefiles = np.array(imagefiles)[np.array(args.ids)]

    for imagefile, in progressbar(imagefiles):
        backendDb.deleteImage(c, imagefile)
