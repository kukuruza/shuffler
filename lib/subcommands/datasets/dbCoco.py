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
from datetime import datetime

from lib.backend import backendDb
from lib.utils import util


def add_parsers(subparsers):
    exportCocoParser(subparsers)


def exportCocoParser(subparsers):
    parser = subparsers.add_parser(
        'exportCoco',
        description='Export the database into COCO format. '
        'Limitations: (1) can only record one lincense, '
        '(2) does not record is_crowd to objects, '
        '(3) does not record supercategory to categories.')
    parser.set_defaults(func=exportCoco)
    parser.add_argument('--coco_dir',
                        required=True,
                        help='Root directory of COCO')
    images_policy = parser.add_mutually_exclusive_group()
    images_policy.add_argument(
        '--copy_images',
        action='store_true',
        help=
        'If specified, will copy images to args.coco_dir/"images"/args.subset. '
        'Required, if the media is stored as video. ')
    images_policy.add_argument(
        '--symlink_images',
        action='store_true',
        help='If specified, creates a symbolic link from imagefiles '
        'to imagefiles at args.coco_dir/"images"/args.subset/. '
        'Valid only if all the media is stored as images.')
    images_policy.add_argument(
        '--symlink_image_folder',
        action='store_true',
        help='If specified, creates a symlink from the folder with images '
        'to args.coco_dir/"images"/args.subset. '
        'Valid only if all media is stored as images in one folder.')
    parser.add_argument('--subset',
                        required=True,
                        help='Name of the subset, such as "train2017')
    parser.add_argument(
        '--categories',
        nargs='+',
        help=
        'If specified, will write only object names in the "categories" list. '
        'If NOT specified, every distinct object name, except NULL, is a category.'
    )
    parser.add_argument('--year',
                        type=int,
                        help='Optional year of the dataset, e.g. 2017')
    parser.add_argument('--version',
                        help='Optional version of the dataset, e.g. "1.0"')
    parser.add_argument(
        '--description',
        help='Optional description of the dataset, e.g. "COCO 2017 Dataset"')
    parser.add_argument(
        '--contributor',
        help='Optional contributor of the dataset, e.g. "COCO Consortium"')
    parser.add_argument(
        '--url',
        help='Optional url of the dataset, e.g. "http://cocodataset.org"')
    parser.add_argument(
        '--date_created',
        help='Optional creation date of the dataset in format YYYY/MM/DD')
    parser.add_argument(
        '--license_url',
        help=
        'Optional license url, e.g. "http://creativecommons.org/licenses/by/2.0/"'
    )
    parser.add_argument(
        '--license_name',
        help='Optional license name, e.g. "Attribution License"')


def exportCoco(c, args):

    # Date created.
    if args.date_created is None:
        date_created = None
    else:
        try:
            date_created = datetime.strptime(args.date_created, '%Y/%m/%d')
        except ValueError as e:
            raise ValueError('date_created is expected in the form YYYY/MM/DD')
    date_created_str = (datetime.strftime(date_created, '%Y/%m/%d')
                        if date_created is not None else None)

    # Info.
    info = {
        "year": args.year,
        "version": args.version,
        "description": args.description,
        "contributor": args.contributor,
        "url": args.url,
        "date_created": date_created_str
    }

    # Licenses.
    if args.license_url is None and args.license_name is None:
        licenses = []
    else:
        license_url = args.license_url if args.license_url is not None else ""
        license_name = args.license_name if args.license_name is not None else ""
        licenses = [{"url": license_url, "id": 1, "name": license_name}]

    # Images.
    new_image_dir = op.join(args.coco_dir, 'images', args.subset)
    if not op.exists(op.dirname(new_image_dir)):
        os.makedirs(op.dirname(new_image_dir))
    if not args.symlink_image_folder and not op.exists(new_image_dir):
        os.makedirs(new_image_dir)

    logging.info('Writing images.')
    images = []  # big list of images.
    imageids = {}  # necessary to retrieve imageid by imagefile.
    c.execute('SELECT * FROM images')
    for imageid, image_entry in progressbar(enumerate(c.fetchall())):
        imagefile = backendDb.imageField(image_entry, 'imagefile')
        width = backendDb.imageField(image_entry, 'width')
        height = backendDb.imageField(image_entry, 'height')
        timestamp = backendDb.imageField(image_entry, 'timestamp')
        file_name = op.basename(imagefile)
        license = 1 if len(
            licenses) == 1 else None  # Only one license is supported.
        date_captured = datetime.strftime(
            backendDb.parseTimeString(timestamp),
            '%Y-%m-%d %H:%M:%S') if timestamp is not None else None
        # For reference in objects.
        imageids[imagefile] = imageid
        # Add to the big list of images.
        images.append({
            "id": imageid,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": license,
            "date_captured": date_captured,
        })

        # Maybe copy or make a symlink for images.
        old_image_path = op.join(args.rootdir, imagefile)
        if not op.exists(old_image_path):
            raise FileNotFoundError(
                "Image not found at %s. (Using rootdir %s.)" %
                (old_image_path, args.rootdir))
        new_image_path = op.join(new_image_dir, file_name)
        if args.copy_images:
            shutil.copyfile(old_image_path, new_image_path)
        elif args.symlink_images:
            os.symlink(op.abspath(old_image_path),
                       new_image_path,
                       target_is_directory=False)
        elif args.symlink_image_folder and not op.exists(new_image_dir):
            old_image_dir = op.dirname(old_image_path)
            os.symlink(op.abspath(old_image_dir),
                       new_image_dir,
                       target_is_directory=True)

    # Categories.
    categories_coco = []  # big list of categories.
    categoryids = {}  # necessary to retrieve categoryid by category name.
    if args.categories is not None:
        categories = args.categories
    else:
        c.execute('SELECT DISTINCT(name) FROM objects WHERE name IS NOT NULL')
        categories = [c for c, in c.fetchall()]
    for icategory, category in enumerate(categories):
        categoryids[category] = icategory
        categories_coco.append({
            "supercategory": None,
            "id": icategory,
            "name": category
        })
    logging.info('Found %d categories: %s' % (len(categoryids), categories))

    # Objects.
    logging.info('Writing objects.')
    annotations = []  # big list of images.
    categories_str = ', '.join(['"%s"' % c for c in categories])
    logging.debug('Will execute: "SELECT * FROM objects WHERE name IN (%s)"',
                  categories_str)
    c.execute('SELECT * FROM objects WHERE name IN (%s)' % categories_str)
    for object_entry in progressbar(c.fetchall()):
        objectid = backendDb.objectField(object_entry, 'objectid')
        name = backendDb.objectField(object_entry, 'name')
        imagefile = backendDb.objectField(object_entry, 'imagefile')
        bbox = backendDb.objectField(object_entry, 'bbox')
        imageid = imageids[imagefile]
        util.polygons2bboxes(c, objectid)  # Write proper bboxes.

        # Get polygons.
        c.execute('SELECT DISTINCT(name) FROM polygons WHERE objectid=?',
                  (objectid, ))
        polygon_names = c.fetchall()
        polygons_coco = []
        for polygon_name, in polygon_names:
            if polygon_name is None:
                c.execute(
                    'SELECT x,y FROM polygons WHERE objectid=? AND name IS NULL',
                    (objectid, ))
            else:
                c.execute(
                    'SELECT x,y FROM polygons WHERE objectid=? AND name=?',
                    (objectid, polygon_name))
            polygon = c.fetchall()
            polygon_coco = np.array(
                polygon).flatten().tolist()  # to [x1, y1, x2, y2, ... xn, yn].
            polygons_coco.append(polygon_coco)

        # Get area.
        mask = util.polygons2mask(c, objectid)
        area = np.count_nonzero(mask)

        annotations.append({
            "id": objectid,
            "image_id": imageid,
            "category_id": categoryids[name],
            "segmentation": polygons_coco,
            "area": area,
            "bbox": list(bbox),
            "iscrowd": 0,
        })

    data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories_coco,
        "licenses": licenses,
    }

    json_path = op.join(args.coco_dir, 'annotations',
                        'instances_%s.json' % args.subset)
    if not op.exists(op.dirname(json_path)):
        os.makedirs(op.dirname(json_path))
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
