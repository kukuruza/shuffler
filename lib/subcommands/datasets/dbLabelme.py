import os, sys, os.path as op
import numpy as np
import cv2
from lxml import etree as ET
import collections
import logging
from glob import glob
import shutil
import sqlite3
from progressbar import progressbar
from pprint import pformat
import re
from datetime import datetime

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util


def add_parsers(subparsers):
    importLabelmeParser(subparsers)
    importLabelmeObjectsParser(subparsers)
    exportLabelmeParser(subparsers)


def _pointsOfPolygon(annotation):
    pts = annotation.find('polygon').findall('pt')
    xs = []
    ys = []
    for pt in pts:
        xs.append(int(float(pt.find('x').text)) - 1)
        ys.append(int(float(pt.find('y').text)) - 1)
    logging.debug('Parsed polygon xs=%s, ys=%s.' % (xs, ys))
    return xs, ys


def _isPolygonDegenerate(xs, ys):
    assert len(xs) == len(ys), (len(xs), len(ys))
    return len(xs) == 1 or len(xs) == 2 or min(xs) == max(xs) or min(
        ys) == max(ys)


def importLabelmeParser(subparsers):
    parser = subparsers.add_parser(
        'importLabelme', description='Import LabelMe annotations for a db.')
    parser.set_defaults(func=importLabelme)
    parser.add_argument('--images_dir',
                        required=True,
                        help='Directory with jpg files.')
    parser.add_argument('--annotations_dir',
                        required=True,
                        help='Directory with xml files.')
    parser.add_argument('--replace',
                        action='store_true',
                        help='Will replace objects with the same objectid'
                        ' as the id in the xml, but keep the same imagefile.')
    parser.add_argument('--with_display', action='store_true')


def importLabelme(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(args.rootdir)

    # Adding images.
    image_paths = (glob(op.join(args.images_dir, '*.jpg')) +
                   glob(op.join(args.images_dir, '*.JPG')))
    logging.info('Adding %d images.' % len(image_paths))
    imagefiles = []
    for image_path in progressbar(image_paths):
        height, width = backendMedia.getPictureSize(image_path)
        imagefile = op.relpath(op.abspath(image_path), args.rootdir)
        timestamp = backendDb.makeTimeString(datetime.now())
        if not args.replace:
            c.execute(
                'INSERT INTO images('
                'imagefile, width, height, timestamp) VALUES (?,?,?,?)',
                (imagefile, width, height, timestamp))
        imagefiles.append(imagefile)
    logging.info('Found %d new imagefiles.' % len(imagefiles))

    # Adding annotations.
    annotations_paths = os.listdir(args.annotations_dir)

    for imagefile in progressbar(imagefiles):
        logging.info('Processing imagefile: "%s"' % imagefile)

        # Find annotation files that match the imagefile.
        # There may be 1 or 2 dots in the extension because of some
        # bug/feature in LabelMe.
        regex = re.compile('0*%s[\.]{1,2}xml' %
                           op.splitext(op.basename(imagefile))[0])
        logging.debug('Will try to match %s' % regex)
        matches = [f for f in annotations_paths if re.match(regex, f)]
        if len(matches) == 0:
            logging.debug('Annotation file does not exist: "%s". Skip image.' %
                          imagefile)
            continue
        elif len(matches) > 1:
            logging.warning('Found multiple files: %s' % pformat(matches))
        # FIXME: pick the latest as opposed to just one of those.
        annotation_file = op.join(args.annotations_dir, matches[0])
        logging.debug('Got a match %s' % annotation_file)

        if args.with_display:
            img = imreader.imread(imagefile)

        tree = ET.parse(annotation_file)
        for object_ in tree.getroot().findall('object'):

            # skip if it was deleted
            if object_.find('deleted').text == '1':
                continue

            # find the name of object.
            name = object_.find('name').text
            if name is not None:
                name = name.encode('utf-8')

            # get all the points
            xs, ys = _pointsOfPolygon(object_)

            # filter out degenerate polygons
            if _isPolygonDegenerate(xs, ys):
                logging.warning('Degenerate polygon %s,%s in %s' %
                                (str(xs), str(ys), annotation_file))
                continue

            if args.replace:
                # If replace, expect the objectid to be there.
                xml_objectid = int(object_.find('id').text)
                c.execute('SELECT COUNT(1) FROM objects WHERE objectid=?',
                          (xml_objectid, ))
                if not c.fetchone()[0]:
                    raise ValueError('"replace" is specified but objectid %d '
                                     'does not exist.' % xml_objectid)
                c.execute(
                    'UPDATE objects SET name=?,x1=NULL,'
                    'y1=NULL,width=NULL,height=NULL WHERE objectid=?',
                    (name, xml_objectid))
                c.execute('DELETE FROM polygons WHERE objectid=?',
                          (xml_objectid, ))
                objectid = xml_objectid
                logging.info('Updated objectid %d.', objectid)
            else:
                c.execute('INSERT INTO objects(imagefile,name) VALUES (?,?)',
                          (imagefile, name))
                objectid = c.lastrowid
                logging.debug('Will assign a new objectid %d.', objectid)

            for i in range(len(xs)):
                c.execute('INSERT INTO polygons(objectid,x,y) VALUES (?,?,?);',
                          (objectid, xs[i], ys[i]))

            if (object_.find('occluded') is not None
                    and object_.find('occluded').text == 'yes'):
                c.execute(
                    'INSERT INTO properties(objectid,key,value) VALUES (?,?,?);',
                    (objectid, 'occluded', 'true'))

            for attrib in object_.findall('attributes'):
                if not attrib.text:
                    continue
                c.execute(
                    'INSERT INTO properties(objectid,key,value) VALUES (?,?,?);',
                    (objectid, attrib.text.encode('utf-8'), 'true'))

            util.polygons2bboxes(c, objectid)  # Generate a bounding box.

            if args.with_display:
                pts = np.array([xs, ys], dtype=np.int32).transpose()
                util.drawScoredPolygon(img, pts, name, score=1)

        if args.with_display:
            cv2.imshow('importLabelmeImages', img[:, :, ::-1])
            if cv2.waitKey(-1) == 27:
                args.with_display = False
                cv2.destroyWindow('importLabelmeImages')


def importLabelmeObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'importLabelmeObjects',
        description='Import LabelMe annotations of objects. For each objectid '
        'in the db, will look for annotation in the form objectid.xml')
    parser.set_defaults(func=importLabelmeObjects)
    parser.add_argument('--annotations_dir',
                        required=True,
                        help='Directory with xml files.')
    parser.add_argument('--with_display', action='store_true')
    parser.add_argument('--keep_original_object_name',
                        action='store_true',
                        help='Do not update the object name from parsed xml.')
    parser.add_argument(
        '--polygon_name',
        help='If specified, give each polygon entry this name.')


def importLabelmeObjects(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    annotations_paths = os.listdir(args.annotations_dir)

    c.execute('SELECT objectid,imagefile FROM objects')
    for objectid, imagefile in progressbar(c.fetchall()):
        logging.debug('Processing object: %d' % objectid)

        # Find annotation files that match the object.
        # There may be 1 or 2 dots because of some bug/feature in LabelMe.
        regex = re.compile('0*%s[\.]{1,2}xml' % str(objectid))
        logging.debug('Will try to match %s' % regex)
        matches = [f for f in annotations_paths if re.match(regex, f)]
        if len(matches) == 0:
            logging.info('Annotation file does not exist: "%s". Skip image.',
                         annotation_file)
            continue
        elif len(matches) > 1:
            logging.warning('Found multiple files: %s', pformat(matches))
        annotation_file = op.join(args.annotations_dir, matches[0])
        logging.info('Got a match %s' % annotation_file)

        tree = ET.parse(annotation_file)
        objects_ = tree.getroot().findall('object')

        # remove all deleted
        objects_ = [
            object_ for object_ in objects_
            if object_.find('deleted').text != '1'
        ]
        if len(objects_) > 1:
            logging.error('More than one object in %s' % annotation_file)
            continue
        object_ = objects_[0]

        # find the name of object.
        name = object_.find('name').text.encode('utf-8')
        if not args.keep_original_object_name:
            c.execute('UPDATE objects SET name=? WHERE objectid=?',
                      (name, objectid))

        if object_.find('occluded').text == 'yes':
            c.execute(
                'INSERT INTO properties(objectid,key,value) VALUES (?,?,?);',
                (objectid, 'occluded', 'true'))

        for attrib in object_.findall('attributes'):
            if not attrib.text:
                continue
            c.execute(
                'INSERT INTO properties(objectid,key,value) VALUES (?,?,?);',
                (objectid, attrib.text.encode('utf-8'), 'true'))

        # get all the points
        xs, ys = _pointsOfPolygon(object_)

        # Filter out degenerate polygons
        if _isPolygonDegenerate(xs, ys):
            logging.error('degenerate polygon %s,%s in %s' %
                          (str(xs), str(ys), annotation_file))
            continue

        # Update polygon.
        for i in range(len(xs)):
            polygon = (objectid, xs[i], ys[i], args.polygon_name)
            c.execute(
                'INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)',
                polygon)

        if args.with_display:
            img = imreader.imread(imagefile)
            pts = np.array([xs, ys], dtype=np.int32).transpose()
            util.drawScoredPolygon(img, pts, name, score=1)
            cv2.imshow('importLabelmeObjects', img[:, :, ::-1])
            if cv2.waitKey(-1) == 27:
                args.with_display = False
                cv2.destroyWindow('importLabelmeObjects')


def exportLabelmeParser(subparsers):
    parser = subparsers.add_parser(
        'exportLabelme', description='Import LabelMe annotations for a db.')
    parser.set_defaults(func=exportLabelme)
    parser.add_argument('--images_dir',
                        help='Directory to write jpg files to. '
                        'If not specified, will not write jpg.')
    parser.add_argument('--annotations_dir',
                        required=True,
                        help='Directory to write xml files to.')
    parser.add_argument('--username',
                        help='Optional LabelMe username. '
                        'If left blank and if polygons have names,')
    parser.add_argument(
        '--folder',
        required=True,
        help='The folder name as it will be called in LabelMeAnnotationTool')
    parser.add_argument('--source_image',
                        default='',
                        help='Optional field to fill in the annotation files.')
    parser.add_argument('--source_annotation',
                        default='LabelMe Webtool',
                        help='Optional field to fill in the annotation files.')
    parser.add_argument(
        '--fix_invalid_image_names',
        action='store_true',
        help='Some symbols are invalid in image names for Labelme. '
        'They will be replaced with "_".')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite image and/or annotation files if they exist.')


def exportLabelme(c, args):
    print_warning_for_multiple_polygons_in_the_end = False

    if not op.exists(args.annotations_dir):
        os.makedirs(args.annotations_dir)

    if args.images_dir is not None:
        imreader = backendMedia.MediaReader(rootdir=args.rootdir)
        imwriter = backendMedia.MediaWriter(media_type='pictures',
                                            rootdir=args.rootdir,
                                            image_media=args.images_dir,
                                            overwrite=args.overwrite)

    c.execute('SELECT imagefile,height,width,timestamp FROM images')
    for imagefile, imheight, imwidth, timestamp in progressbar(c.fetchall()):

        el_root = ET.Element("annotation")
        ET.SubElement(el_root, "filename").text = os.path.basename(imagefile)
        ET.SubElement(el_root, "folder").text = args.folder

        el_source = ET.SubElement(el_root, "source")
        ET.SubElement(el_source, 'sourceImage').text = args.source_image
        ET.SubElement(el_source,
                      'sourceAnnotation').text = args.source_annotation

        el_imagesize = ET.SubElement(el_root, "imagesize")
        ET.SubElement(el_imagesize, 'nrows').text = str(imheight)
        ET.SubElement(el_imagesize, 'ncols').text = str(imwidth)

        time = backendDb.parseTimeString(timestamp)
        timestamp = datetime.strftime(time, '%Y-%m-%d %H:%M:%S')

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        for object_entry in c.fetchall():
            objectid = backendDb.objectField(object_entry, 'objectid')
            name = backendDb.objectField(object_entry, 'name')
            c.execute(
                'SELECT value FROM properties '
                'WHERE objectid = ? AND key = "occluded"', (objectid, ))
            occluded = c.fetchone()
            # The object is occluded if "occluded" is a recorded property
            # and its value is "true".
            occluded = 'yes' if occluded is not None and occluded == 'true' else 'no'

            # In case bboxes were not recorded as polygons.
            util.bboxes2polygons(c, objectid)

            el_object = ET.SubElement(el_root, "object")
            ET.SubElement(el_object,
                          'name').text = name if name is not None else 'object'
            ET.SubElement(el_object, 'deleted').text = '0'
            ET.SubElement(el_object, 'verified').text = '0'
            ET.SubElement(el_object, 'type').text = 'bounding_box'
            ET.SubElement(el_object, 'id').text = str(objectid)
            ET.SubElement(el_object, 'date').text = timestamp
            ET.SubElement(el_object, 'occluded').text = occluded

            # Parts.
            el_parts = ET.SubElement(el_object, 'parts')
            ET.SubElement(el_parts, 'hasparts')
            ET.SubElement(el_parts, 'ispartof')

            # Attributes.
            c.execute('SELECT key FROM properties WHERE objectid=?',
                      (objectid, ))
            for key, in c.fetchall():
                if not isinstance(key, str):
                    key = key.decode("utf-8")
                ET.SubElement(el_object, 'property').text = key

            # Polygons.
            c.execute('SELECT DISTINCT(name) FROM polygons WHERE objectid=?',
                      (objectid, ))
            pol_names = [x for x, in c.fetchall()]
            logging.debug('For objectid %d found %d polygons' %
                          (objectid, len(pol_names)))
            if len(pol_names) > 1:
                print_warning_for_multiple_polygons_in_the_end = True
                logging.warning(
                    'objectid %d has multiple polygons: %s. Wrote only the first.'
                    % (objectid, pformat(pol_names)))
            for pol_name in pol_names:
                el_polygon = ET.SubElement(el_object, 'polygon')
                # Recording the username.
                if args.username is not None:
                    username = args.username
                elif pol_name is not None:
                    username = pol_name
                else:
                    username = 'anonymous'
                ET.SubElement(el_polygon, 'username').text = username
                # Recording points.
                if pol_name is None:
                    c.execute(
                        'SELECT x,y FROM polygons '
                        'WHERE objectid=? AND name IS NULL', (objectid, ))
                else:
                    c.execute(
                        'SELECT x,y FROM polygons '
                        'WHERE objectid=? AND name=?', (objectid, pol_name))
                xy_entries = c.fetchall()
                for x, y in xy_entries:
                    el_point = ET.SubElement(el_polygon, 'pt')
                    ET.SubElement(el_point, 'x').text = str(x)
                    ET.SubElement(el_point, 'y').text = str(y)
                logging.debug('Polygon %s has %d points.' %
                              (pol_name, len(xy_entries)))

        # Get and maybe fix imagename.
        imagename = op.basename(imagefile)
        if args.fix_invalid_image_names:
            imagename_fixed = re.sub(r'[^a-zA-Z0-9_.-]', '_', imagename)
            if imagename_fixed != imagename:
                imagename = imagename_fixed
                logging.warning('Replaced invalid characters in image name %s',
                                imagename)

        logging.debug('Writing imagefile %s as:\n%s', imagefile,
                      ET.tostring(el_root, pretty_print=True).decode("utf-8"))

        # Write annotation.
        annotation_name = '%s.xml' % op.splitext(imagename)[0]
        annotation_path = op.join(args.annotations_dir, annotation_name)
        logging.debug('Will write annotation to "%s"' % annotation_path)
        if op.exists(annotation_path) and not args.overwrite:
            raise FileExistsError('Annotation file "%s" already exists. '
                                  'Maybe pass "overwrite" argument.')
        with open(annotation_path, 'wb') as f:
            f.write(ET.tostring(el_root, pretty_print=True))

        # Write image.
        if args.images_dir is not None:
            image = imreader.imread(imagefile)
            imagefile = imwriter.imwrite(image, namehint=imagename)

    if print_warning_for_multiple_polygons_in_the_end:
        logging.warning(
            'There were multiple polygons for some object. '
            'Multiple polygons are not supported in LabelMe: '
            'https://github.com/wkentaro/labelme/issues/35. '
            'We wrote all polygons but probably only one will be used by LabelMe.'
        )
