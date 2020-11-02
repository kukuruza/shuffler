import os, sys, os.path as op
import numpy as np
import cv2
import logging
from glob import glob
import shutil
import simplejson as json
import sqlite3
from progressbar import progressbar
from pprint import pformat

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util


def add_parsers(subparsers):
    importCityscapesParser(subparsers)
    exportCityscapesParser(subparsers)


def _importJson(c, jsonpath, imagefile, imheight, imwidth):
    ''' Imports one file with polygons of Cityscapes format. '''

    with open(jsonpath) as f:
        image_dict = json.load(f)
    if image_dict['imgHeight'] != imheight or image_dict['imgWidth'] != imwidth:
        raise Exception('Image size is inconsistent to json: %dx%d vs %dx%d' %
                        (image_dict['imgHeight'], image_dict['imgWidth'],
                         imheight, imwidth))
    objects = image_dict['objects']
    for object_ in objects:
        name = object_['label']
        polygon = object_['polygon']
        x1 = int(np.min([point[0] for point in polygon]))
        y1 = int(np.min([point[1] for point in polygon]))
        x2 = int(np.max([point[0] for point in polygon]))
        y2 = int(np.max([point[1] for point in polygon]))
        width = x2 - x1
        height = y2 - y1
        c.execute(
            'INSERT INTO objects(imagefile,x1,y1,width,height,name) '
            'VALUES (?,?,?,?,?,?)', (imagefile, x1, y1, width, height, name))
        objectid = c.lastrowid
        for point in polygon:
            s = 'INSERT INTO polygons(objectid,x,y) VALUES (?,?,?)'
            c.execute(s, (objectid, point[0], point[1]))


def importCityscapesParser(subparsers):
    parser = subparsers.add_parser(
        'importCityscapes',
        description='Import Cityscapes annotations into the database. '
        'Images are assumed to be from leftImg8bit. For the CityScapes format, '
        'please visit https://github.com/mcordts/cityscapesScripts.')
    parser.set_defaults(func=importCityscapes)
    parser.add_argument(
        '--cityscapes_dir',
        required=True,
        help='Root directory of Cityscapes. It should contain subdirs '
        '"gtFine_trainvaltest", "leftImg8bit_trainvaltest", etc.')
    parser.add_argument(
        '--type',
        default='gtFine',
        choices=['gtFine', 'gtCoarse'],
        help='Which annotations to parse. '
        'Will not parse both to keep the logic straightforward.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test', 'val'],
        choices=['train', 'test', 'val', 'train_extra', 'demoVideo'],
        help='Splits to be parsed.')
    parser.add_argument('--mask_type',
                        choices=['labelIds', 'instanceIds', 'color'],
                        help='Which mask to import, if any.')
    parser.add_argument('--with_display', action='store_true')


def importCityscapes(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(args.rootdir)

    logging.info('Will load splits: %s' % args.splits)
    logging.info('Will load json type: %s' % args.type)
    logging.info('Will load mask type: %s' % args.mask_type)

    if not op.exists(args.cityscapes_dir):
        raise Exception('Cityscape directory "%s" does not exist' %
                        args.cityscapes_dir)

    # Directory accessed by label type and by split.
    dirs_by_typesplit = {}

    for type_ in [args.type, 'leftImg8bit']:
        type_dir_template = op.join(args.cityscapes_dir, '%s*' % type_)
        for type_dir in [x for x in glob(type_dir_template) if op.isdir(x)]:
            logging.debug('Looking for splits in %s' % type_dir)
            for split in args.splits:
                typesplit_dir = op.join(type_dir, split)
                if op.exists(typesplit_dir):
                    logging.debug('Found split %s in %s' % (split, type_dir))
                    # Add the info into the main dictionary "dirs_by_typesplit".
                    if split not in dirs_by_typesplit:
                        dirs_by_typesplit[split] = {}
                    dirs_by_typesplit[split][type_] = typesplit_dir
    logging.info('Found the following directories: \n%s' %
                 pformat(dirs_by_typesplit))

    for split in args.splits:
        # List of cities.
        assert 'leftImg8bit' in dirs_by_typesplit[split]
        leftImg8bit_dir = dirs_by_typesplit[split]['leftImg8bit']
        cities = os.listdir(leftImg8bit_dir)
        cities = [x for x in cities if op.isdir(op.join(leftImg8bit_dir, x))]
        logging.info('Found %d cities in %s' % (len(cities), leftImg8bit_dir))

        for city in cities:
            image_names = os.listdir(op.join(leftImg8bit_dir, city))
            logging.info('In split "%s", city "%s" has %d images' %
                         (split, city, len(image_names)))

            for image_name in image_names:

                # Get the image path.
                image_path = op.join(leftImg8bit_dir, city, image_name)
                name_parts = op.splitext(image_name)[0].split('_')
                if len(name_parts) != 4:
                    raise Exception(
                        'Expect to have 4 parts in the name of image "%s"' %
                        image_name)
                if name_parts[0] != city:
                    raise Exception('The first part of name of image "%s" '
                                    'is expected to be city "%s".' %
                                    (image_name, city))
                if name_parts[3] != 'leftImg8bit':
                    raise Exception('The last part of name of image "%s" '
                                    'is expected to be city "leftImg8bit".' %
                                    image_name)
                imheight, imwidth = backendMedia.getPictureSize(image_path)
                imagefile = op.relpath(image_path, args.rootdir)
                c.execute(
                    'INSERT INTO images(imagefile,width,height) VALUES (?,?,?)',
                    (imagefile, imwidth, imheight))

                # Get the json label.
                if args.type in dirs_by_typesplit[split]:
                    city_dir = op.join(dirs_by_typesplit[split][args.type],
                                       city)
                    if op.exists(city_dir):

                        json_name = '_'.join(name_parts[:3] +
                                             [args.type, 'polygons']) + '.json'
                        json_path = op.join(city_dir, json_name)
                        if op.exists(json_path):
                            _importJson(c, json_path, imagefile, imheight,
                                        imwidth)

                        if args.mask_type is not None:
                            mask_name = '_'.join(
                                name_parts[:3] +
                                [args.type, args.mask_type]) + '.png'
                            mask_path = op.join(city_dir, mask_name)
                            if op.exists(mask_path):
                                maskfile = op.relpath(mask_path, args.rootdir)
                                c.execute(
                                    'UPDATE images SET maskfile=? WHERE imagefile=?',
                                    (maskfile, imagefile))

                if args.with_display:
                    img = imreader.imread(imagefile)
                    c.execute(
                        'SELECT objectid,name FROM objects WHERE imagefile=?',
                        (imagefile, ))
                    for objectid, name in c.fetchall():
                        # Draw polygon.
                        c.execute('SELECT x,y FROM polygons WHERE objectid=?',
                                  (objectid, ))
                        polygon = c.fetchall()
                        util.drawScoredPolygon(img, [(int(pt[0]), int(pt[1]))
                                                     for pt in polygon], name)
                    cv2.imshow('importCityscapes', img[:, :, ::-1])
                    if cv2.waitKey(-1) == 27:
                        args.with_display = False
                        cv2.destroyWindow('importCityscapes')

    # Statistics.
    c.execute('SELECT COUNT(1) FROM images')
    logging.info('Imported %d images' % c.fetchone()[0])
    c.execute('SELECT COUNT(1) FROM images WHERE maskfile IS NOT NULL')
    logging.info('Imported %d masks' % c.fetchone()[0])
    c.execute('SELECT COUNT(DISTINCT(imagefile)) FROM objects')
    logging.info('Objects are found in %d images' % c.fetchone()[0])


def _makeLabelName(name, city, type_, was_imported_from_cityscapes):
    '''
  If a name is like {X}_{Y}_{Z}, then city_{Y}_type.
  If a name is like {Y}, then city_{Y}_type.
  '''
    # Drop extension.
    core = op.splitext(name)[0]
    # Split into parts, and put them together with correct city and type.
    if was_imported_from_cityscapes:
        core_parts = core.split('_')
        if len(core_parts) >= 3:
            core = '_'.join(core_parts[1:-1])
    out_name = '_'.join([city, core, type_])
    logging.debug('Made name "%s" out of input name "%s".' % (out_name, name))
    return out_name


def _exportLabel(c, out_path_noext, imagefile):
    '''
    Writes objects to json file.
    Please use https://github.com/mcordts/cityscapesScripts to generate masks.
    '''
    c.execute('SELECT width,height FROM images WHERE imagefile=?',
              (imagefile, ))
    imgWidth, imgHeight = c.fetchone()

    data = {'imgWidth': imgWidth, 'imgHeight': imgHeight, 'objects': []}

    c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
    for object_ in c.fetchall():
        objectid = backendDb.objectField(object_, 'objectid')
        name = backendDb.objectField(object_, 'name')

        # Polygon = [[x1, y1], [x2, y2], ...].
        # Check if the object has a polygon.
        c.execute('SELECT x,y FROM polygons WHERE objectid=?', (objectid, ))
        polygon = c.fetchall()
        if len(polygon) == 0:
            # If there is no polygon, make one from bounding box.
            [y1, x1, y2, x2] = backendDb.objectField(object_, 'roi')
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            color = (255, )
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)

        data['objects'].append({'label': name, 'polygon': polygon})

    json_path = '%s_polygons.json' % out_path_noext
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def exportCityscapesParser(subparsers):
    parser = subparsers.add_parser(
        'exportCityscapes',
        description='Export the database into Cityscapes format. '
        'Export to a single annotations <type>. Furthermore, export to a '
        'single <split> and <city> or infer them from the folder structure. '
        'If an object has no polygons, a rectangular polygon is generated from '
        'its bounding box. To generate masks from polygons, use the official '
        'tool from CityScapes: https://github.com/mcordts/cityscapesScripts')
    parser.set_defaults(func=exportCityscapes)
    parser.add_argument('--cityscapes_dir',
                        required=True,
                        help='Root directory of Cityscapes')
    parser.add_argument(
        '--image_type',
        default='leftImg8bit',
        help='If specified, images will be copied to that folder.'
        'If NOT specified, images will not be exported in any way.')
    parser.add_argument(
        '--type',
        default='gtFine',
        help='Will use this type name. Cityscapes uses "gtFine", "gtCoarse".')
    parser.add_argument(
        '--split',
        help='If specified, will use this split name. '
        'Cityscapes uses "train", "val", "test". If NOT specified, will infer '
        'from the grandparent directory of each imagefile.')
    parser.add_argument(
        '--city',
        help='If specified, only that city will be used. If NOT specified, '
        'will infer from the parent directory of each imagefile.')
    parser.add_argument(
        '--was_imported_from_cityscapes',
        action='store_true',
        help='"Imagefiles" with underscore are parsed as '
        '\{city\}_\{name\}_\{type\}\{ext\}, '
        '\{City\} and \{type\} are changed according to the export settings.')


def exportCityscapes(c, args):
    c.execute('SELECT imagefile FROM images')
    for imagefile, in progressbar(c.fetchall()):

        # Parse the image name, city, and split.
        imagename = op.basename(imagefile)
        city = args.city if args.city is not None else op.basename(
            op.dirname(imagefile))
        split = args.split if args.split is not None else op.basename(
            op.dirname(op.dirname(imagefile)))
        logging.debug('Will write imagefile %s to split %s and city %s.' %
                      (imagefile, split, city))
        out_name = _makeLabelName(imagename, city, args.type,
                                  args.was_imported_from_cityscapes)

        # Write image.
        if args.image_type:
            in_imagepath = op.join(args.rootdir, imagefile)
            if not op.exists(in_imagepath):
                raise Exception(
                    'Problem with the database: image does not exist at %s. '
                    'Check that imagefile refers to an image (not a video frame).'
                    % in_imagepath)
            # TODO: do something if the input is video.

            out_imagedir = op.join(args.cityscapes_dir, args.image_type, split,
                                   city)
            if not op.exists(out_imagedir):
                os.makedirs(out_imagedir)

            out_imagepath = op.join(out_imagedir, '%s.png' % out_name)

            shutil.copyfile(in_imagepath, out_imagepath)
            # TODO: do something if the inout is not png.

        # Write label.
        out_labeldir = op.join(args.cityscapes_dir, args.type, split, city)
        if not op.exists(out_labeldir):
            os.makedirs(out_labeldir)

        out_path_noext = op.join(out_labeldir, out_name)
        _exportLabel(c, out_path_noext, imagefile)
