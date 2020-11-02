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

from lib.backend import backendMedia
from lib.utils import util
from lib.utils import utilBoxes


def add_parsers(subparsers):
    importPascalVoc2012Parser(subparsers)


def importPascalVoc2012Parser(subparsers):
    parser = subparsers.add_parser(
        'importPascalVoc2012',
        description='Import annotations in PASCAL VOC 2012 format into a db. '
        'Object parts are imported as polygons. Either SegmentationClass or '
        'SegmentationObject (not both) can be imported.')
    parser.set_defaults(func=importPascalVoc2012)
    parser.add_argument(
        'pascal_dir',
        help='Path to directory with subdirs "JPEGImages", "Annotations", '
        '"SegmentationClass", and "SegmentationObject". '
        'E.g. "/my/dir/to/VOC2012"')
    parser.add_argument('--annotations',
                        action='store_true',
                        help='Import from directory "Annotations"')
    segm = parser.add_mutually_exclusive_group()
    segm.add_argument('--segmentation_class',
                      action='store_true',
                      help='Import from directory "SegmentationClass"')
    segm.add_argument('--segmentation_object',
                      action='store_true',
                      help='Import from directory "SegmentationObject"')
    parser.add_argument('--with_display', action='store_true')


def importPascalVoc2012(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(args.rootdir)

    image_paths = sorted(glob(op.join(args.pascal_dir, 'JPEGImages/*.jpg')))
    logging.info('Found %d JPG images in %s/JPEGImages', len(image_paths),
                 args.pascal_dir)

    for image_path in progressbar(image_paths):
        filename = op.splitext(op.basename(image_path))[0]
        logging.debug('Processing image: "%s"' % filename)

        # Add image to the database.
        imheight, imwidth = backendMedia.getPictureSize(image_path)
        imagefile = op.relpath(image_path, args.rootdir)
        c.execute('INSERT INTO images(imagefile,width,height) VALUES (?,?,?)',
                  (imagefile, imwidth, imheight))

        if args.with_display:
            img = imreader.imread(imagefile)

        # Object annotations.
        if args.annotations:
            annotation_path = op.join(args.pascal_dir,
                                      'Annotations/%s.xml' % filename)
            if not op.exists(annotation_path):
                raise FileNotFoundError('Annotation file not found at "%s".' %
                                        annotation_path)

            tree = ET.parse(annotation_path)
            for object_ in tree.getroot().findall('object'):

                name = (object_.find('name').text)
                pose = (object_.find('pose').text
                        if object_.findall('pose') else None)
                truncated = (object_.find('truncated').text
                             if object_.findall('truncated') else None)
                difficult = (object_.find('difficult').text
                             if object_.findall('difficult') else None)
                occluded = (object_.find('occluded').text
                            if object_.findall('occluded') else None)
                x1 = int(float(object_.find('bndbox').find('xmin').text))
                y1 = int(float(object_.find('bndbox').find('ymin').text))
                width = int(float(
                    object_.find('bndbox').find('xmax').text)) - x1 + 1
                height = int(float(
                    object_.find('bndbox').find('ymax').text)) - y1 + 1

                c.execute(
                    'INSERT INTO objects(imagefile,x1,y1,width,height,name) '
                    'VALUES (?,?,?,?,?,?)',
                    (imagefile, x1, y1, width, height, name))
                objectid = c.lastrowid
                s = 'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)'
                c.execute(s, (objectid, 'pose', pose))
                c.execute(s, (objectid, 'truncated', truncated))
                c.execute(s, (objectid, 'difficult', difficult))
                c.execute(s, (objectid, 'occluded', occluded))

                if args.with_display:
                    roi = utilBoxes.bbox2roi((x1, y1, width, height))
                    util.drawScoredRoi(img, roi, name)

                for part in object_.findall('part'):
                    # Each part is inserted as a 4-point polygon with the name
                    # of that part.
                    name = part.find('name').text
                    x1 = int(float(part.find('bndbox').find('xmin').text))
                    y1 = int(float(part.find('bndbox').find('ymin').text))
                    x2 = int(float(part.find('bndbox').find('xmax').text))
                    y2 = int(float(part.find('bndbox').find('ymax').text))
                    logging.debug('Found part "%s"' % name)
                    s = 'INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,?)'
                    c.execute(s, (objectid, x1, y1, name))
                    c.execute(s, (objectid, x1, y2, name))
                    c.execute(s, (objectid, x2, y2, name))
                    c.execute(s, (objectid, x2, y1, name))

                    if args.with_display:
                        pts = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                        util.drawScoredPolygon(img, pts, name, score=1)

        # Class egmentation annotations.
        if args.segmentation_class:
            segmentation_path = op.join(args.pascal_dir,
                                        'SegmentationClass/%s.png' % filename)
            if not op.exists(segmentation_path):
                logging.debug('Annotation file not found at "%s".' %
                              segmentation_path)
            else:
                # Add image to the database.
                maskfile = op.relpath(segmentation_path, args.rootdir)
                c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                          (maskfile, imagefile))
                if args.with_display:
                    mask = imreader.maskread(maskfile)
                    img = util.drawMaskAside(img, mask, labelmap=None)

        # Object segmentation annotations.
        elif args.segmentation_object:
            segmentation_path = op.join(args.pascal_dir,
                                        'SegmentationObject/%s.png' % filename)
            if not op.exists(segmentation_path):
                logging.debug('Annotation file not found at "%s".' %
                              segmentation_path)
            else:
                # Add image to the database.
                maskfile = op.relpath(segmentation_path, args.rootdir)
                c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                          (maskfile, imagefile))
                if args.with_display:
                    mask = imreader.maskread(maskfile)
                    img = util.drawMaskAside(img, mask, labelmap=None)

        # Maybe display.
        if args.with_display:
            img = img[:, :,
                      [2, 1, 0, 3]] if img.shape[2] == 4 else img[:, :, ::-1]
            cv2.imshow('importPascalVoc2012', img)
            if cv2.waitKey(-1) == 27:
                args.with_display = False
                cv2.destroyWindow('importPascalVoc2012')
