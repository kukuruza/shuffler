import os, os.path as op
import sqlite3
from lxml import etree as ET
import logging
from glob import glob
from progressbar import progressbar
from pprint import pformat
import re
from datetime import datetime

from shuffler.backend import backend_db
from shuffler.backend import backend_media
from shuffler.utils import general as general_utils
from shuffler.utils import parser as parser_utils


def add_parsers(subparsers):
    importLabelmeParser(subparsers)
    exportLabelmeParser(subparsers)


def _pointsOfPolygon(annotation):
    pts = annotation.find('polygon').findall('pt')
    xs = []
    ys = []
    for pt in pts:
        try:
            # Labelme does not process float, so round to int.
            xs.append(int(float(pt.find('x').text)))
            ys.append(int(float(pt.find('y').text)))
        except ValueError:
            logging.warning('Failed to parse x=%s or y=%s. Skip point.',
                            pt.find('x').text,
                            pt.find('y').text)
    logging.debug('Parsed polygon xs=%s, ys=%s.', xs, ys)
    return xs, ys


def _isPolygonDegenerate(xs, ys):
    assert len(xs) == len(ys), (len(xs), len(ys))
    return (len(xs) == 0 or len(ys) == 0 or len(xs) == 1 or len(xs) == 2
            or min(xs) == max(xs) or min(ys) == max(ys))


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
    parser.add_argument(
        '--ref_db_file',
        help='If export to labelme was made with shuffler, the original '
        '"imagefile" entries were saved to the "name" field of the database. '
        'Specify ref_db_file to load imagefiles from the "name" field.')


def importLabelme(c, args):
    if args.ref_db_file:
        if not op.exists(args.ref_db_file):
            raise FileNotFoundError('Ref db does not exist: %s' %
                                    args.ref_db_file)
        conn_ref = sqlite3.connect('file:%s?mode=ro' % args.ref_db_file,
                                   uri=True)
        c_ref = conn_ref.cursor()

        # Get imagefiles.
        c_ref.execute('SELECT imagefile,name FROM images')
        image_entries_ref = c_ref.fetchall()
        logging.info('Ref db has %d imagefiles.', len(image_entries_ref))
        labelme_to_original_imagefile_map = {
            imagefile: name
            for (imagefile, name) in image_entries_ref
        }
    else:
        labelme_to_original_imagefile_map = None

    def get_original_imagefile(labelme_imagefile,
                               labelme_to_original_imagefile_map):
        if labelme_to_original_imagefile_map is not None:
            if labelme_imagefile not in labelme_to_original_imagefile_map:
                raise KeyError(
                    'Can not find key "%s" in "name" field in ref_db_file.' %
                    labelme_imagefile)
            return labelme_to_original_imagefile_map[labelme_imagefile]
        else:
            return labelme_imagefile

    # Adding images.
    image_paths = (glob(op.join(args.images_dir, '*.jpg')) +
                   glob(op.join(args.images_dir, '*.JPG')))
    logging.info('Adding %d images.', len(image_paths))
    imagefiles = []
    for image_path in progressbar(image_paths):
        height, width = backend_media.getPictureSize(image_path)
        labelme_imagefile = op.relpath(op.abspath(image_path), args.rootdir)
        original_imagefile = get_original_imagefile(
            labelme_imagefile, labelme_to_original_imagefile_map)
        timestamp = backend_db.makeTimeString(datetime.now())
        if not args.replace:
            c.execute(
                'INSERT INTO images('
                'imagefile, width, height, timestamp) VALUES (?,?,?,?)',
                (original_imagefile, width, height, timestamp))
        imagefiles.append((labelme_imagefile, original_imagefile))
    logging.info('Found %d new imagefiles.', len(imagefiles))

    # Adding annotations.
    annotations_paths = os.listdir(args.annotations_dir)

    for labelme_imagefile, original_imagefile in progressbar(imagefiles):
        logging.info(
            'Processing imagefile: "%s", which has original name "%s"',
            labelme_imagefile, original_imagefile)

        # Find annotation files that match the imagefile.
        # There may be 1 or 2 dots in the extension because of some
        # bug/feature in LabelMe.
        regex = re.compile('0*%s[\.]{1,2}xml' %
                           op.splitext(op.basename(labelme_imagefile))[0])
        logging.debug('Will try to match %s', regex)
        matches = [f for f in annotations_paths if re.match(regex, f)]
        if len(matches) == 0:
            logging.debug('Annotation file does not exist: "%s". Skip image.' %
                          labelme_imagefile)
            continue
        elif len(matches) > 1:
            logging.warning('Found multiple files: %s', pformat(matches))
        # FIXME: pick the latest as opposed to just one of those.
        annotation_file = op.join(args.annotations_dir, matches[0])
        logging.debug('Got a match %s', annotation_file)

        tree = ET.parse(annotation_file)
        for object_ in tree.getroot().findall('object'):

            # skip if it was deleted
            if object_.find('deleted').text == '1':
                continue

            # find the name of object.
            name = object_.find('name').text

            # get all the points
            xs, ys = _pointsOfPolygon(object_)

            # filter out degenerate polygons
            if _isPolygonDegenerate(xs, ys):
                logging.warning('Degenerate polygon %s,%s in %s', str(xs),
                                str(ys), annotation_file)
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
                          (original_imagefile, name))
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
                    (objectid, attrib.text, 'true'))

            general_utils.polygons2bboxes(c,
                                          objectid)  # Generate a bounding box.


def exportLabelmeParser(subparsers):
    parser = subparsers.add_parser(
        'exportLabelme',
        description='Export database in LabelMe format. '
        'Basenames of "imagefile" entries in the db will be used as names of recorded images and annotations.'
    )
    parser.set_defaults(func=exportLabelme)
    parser.add_argument('--images_dir',
                        required=True,
                        help='Directory to write jpg files to. '
                        'If not specified, will not write jpg. '
                        'Normally, it will be "/path/to/labelme/Images".')
    parser.add_argument('--annotations_dir',
                        required=True,
                        help='Directory to write xml files to. '
                        'Normally, it will be "/path/to/labelme/Annotations".')
    parser.add_argument('--username', help='Optional LabelMe username.')
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
        '--overwrite',
        action='store_true',
        help='overwrite image and/or annotation files if they exist.')
    parser_utils.addExportedImageNameArguments(parser)


def exportLabelme(c, args):
    print_warning_for_multiple_polygons_in_the_end = False

    # Check if names of files will be unique.
    c.execute('SELECT imagefile FROM images')
    imagenames = [op.basename(x) for x, in c.fetchall()]
    if len(set(imagenames)) < len(imagenames):
        raise ValueError(
            'Image BASENAMES are not unique in the database. '
            'Currently not able to export because the output files names are '
            'inferred from basenames of "imagefiles".')

    if not op.exists(args.annotations_dir):
        os.makedirs(args.annotations_dir)

    if args.images_dir is not None:
        imreader = backend_media.MediaReader(rootdir=args.rootdir)
        imwriter = backend_media.MediaWriter(media_type='pictures',
                                             rootdir=args.rootdir,
                                             image_media=args.images_dir,
                                             overwrite=args.overwrite)

    c.execute('SELECT imagefile,height,width,timestamp FROM images')
    for imagefile, imheight, imwidth, timestamp in progressbar(c.fetchall()):

        el_root = ET.Element("annotation")
        ET.SubElement(el_root, "folder").text = args.folder

        el_source = ET.SubElement(el_root, "source")
        ET.SubElement(el_source, 'sourceImage').text = args.source_image
        ET.SubElement(el_source,
                      'sourceAnnotation').text = args.source_annotation

        el_imagesize = ET.SubElement(el_root, "imagesize")
        ET.SubElement(el_imagesize, 'nrows').text = str(imheight)
        ET.SubElement(el_imagesize, 'ncols').text = str(imwidth)

        if timestamp is not None:
            time = backend_db.parseTimeString(timestamp)
            timestamp = datetime.strftime(time, '%Y-%m-%d %H:%M:%S')
        else:
            timestamp = ''

        c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
        for object_entry in c.fetchall():
            objectid = backend_db.objectField(object_entry, 'objectid')
            name = backend_db.objectField(object_entry, 'name')
            c.execute(
                'SELECT value FROM properties '
                'WHERE objectid = ? AND key = "occluded"', (objectid, ))
            occluded = c.fetchone()
            # The object is occluded if "occluded" is a recorded property
            # and its value is "true".
            occluded = 'yes' if occluded is not None and occluded == 'true' else 'no'

            # In case bboxes were not recorded as polygons.
            general_utils.bboxes2polygons(c, objectid)

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
            logging.debug('For objectid %d found %d polygons', objectid,
                          len(pol_names))
            if len(pol_names) > 1:
                print_warning_for_multiple_polygons_in_the_end = True
                logging.warning(
                    'objectid %d has multiple polygons: %s. Wrote only the first.',
                    objectid, pformat(pol_names))
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
                logging.debug('Polygon %s has %d points.', pol_name,
                              len(xy_entries))

        c.execute('UPDATE images SET name=? WHERE imagefile=?',
                  (imagefile, imagefile))

        # Write image.
        image_path = general_utils.makeExportedImageName(
            args.images_dir, imagefile, args.dirtree_level_for_name,
            args.fix_invalid_image_names)
        # Labelme supports only JPG.
        if op.splitext(image_path)[1] != '.jpg':
            image_path = op.splitext(image_path)[0] + '.jpg'
            logging.info('Changing the extension to "jpg": %s', image_path)
        logging.debug('Writing imagefile to: %s', image_path)
        image_name = op.basename(image_path)

        # Write image.
        image = imreader.imread(imagefile)
        new_imagefile = imwriter.imwrite(image, namehint=image_name)
        c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, imagefile))
        c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, imagefile))
        ET.SubElement(el_root, "filename").text = os.path.basename(image_name)

        # Write annotation.
        annotation_name = '%s.xml' % op.splitext(image_name)[0]
        annotation_path = op.join(args.annotations_dir, annotation_name)
        logging.debug('Writing annotation to "%s"', annotation_path)
        if op.exists(annotation_path) and not args.overwrite:
            raise FileExistsError('Annotation file "%s" already exists. '
                                  'Maybe pass "overwrite" argument.')
        with open(annotation_path, 'wb') as f:
            f.write(ET.tostring(el_root, pretty_print=True))

    if print_warning_for_multiple_polygons_in_the_end:
        logging.warning(
            'There were multiple polygons for some object. '
            'Multiple polygons are not supported in LabelMe: '
            'https://github.com/wkentaro/labelme/issues/35. '
            'We wrote all polygons but probably only one will be used by LabelMe.'
        )
