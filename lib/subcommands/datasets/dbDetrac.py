import os, sys, os.path as op
import numpy as np
import cv2
import collections
import logging
from glob import glob
from lxml import etree as ET
import shutil
import sqlite3
from progressbar import progressbar
from pprint import pformat

from lib.backend import backendMedia
from lib.utils import util
from lib.utils import utilBoxes


def add_parsers(subparsers):
    importDetracParser(subparsers)


def importDetracParser(subparsers):
    parser = subparsers.add_parser(
        'importDetrac', description='Import DETRAC annotations into a db. ')
    parser.set_defaults(func=importDetrac)
    parser.add_argument('--videos_dir',
                        required=True,
                        help='Directory with .png images. '
                        'E.g. "".')
    parser.add_argument('--tracking_dir',
                        help='Directory with .xml annotations of videos. '
                        'E.g. "detrac/DETRAC-Train-Annotations-XML".')
    parser.add_argument('--with_display', action='store_true')


def importDetrac(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(args.rootdir)

    video_dirs = sorted(glob(op.join(args.videos_dir, 'MVI_?????')))
    logging.info('Found %d videos in %s' % (len(video_dirs), args.videos_dir))

    for video_dir in progressbar(video_dirs):
        video_name = op.splitext(op.basename(video_dir))[0]
        logging.debug('Video name: %s' % video_name)

        # Add images to the db.
        image_paths = glob(op.join(video_dir, '*.jpg'))
        for image_path in image_paths:
            logging.debug('Processing image: "%s"' % image_path)
            if not op.exists(image_path):
                raise FileNotFoundError('Image file not found at "%s".' %
                                        image_path)

            imheight, imwidth = backendMedia.getPictureSize(image_path)
            imagefile = op.relpath(image_path, args.rootdir)
            logging.debug(
                'Parsed image into imagefile=%s, width=%s, height=%s' %
                (imagefile, imwidth, imheight))
            c.execute(
                'INSERT INTO images(imagefile,width,height) VALUES (?,?,?)',
                (imagefile, imwidth, imheight))

            # Maybe display if there are no annotations.
            if args.with_display and args.tracking_dir is None:
                img = imreader.imread(imagefile)
                cv2.imshow('importDetrac', img[:, :, ::-1])
                if cv2.waitKey(-1) == 27:
                    args.with_display = False
                    cv2.destroyWindow('importDetrac')

        # Detection annotations.
        if args.tracking_dir is not None:
            annotation_path = op.join(args.tracking_dir, '%s.xml' % video_name)
            if not op.exists(annotation_path):
                raise FileNotFoundError('Annotation file not found at "%s".' %
                                        annotation_path)

        # For every video, match ids start from 0.
        c.execute('SELECT MAX(match) FROM matches')
        matches_offset = c.fetchone()[0]
        if matches_offset is None:
            matches_offset = 0

        # Read annotation file.
        if args.tracking_dir is not None:
            tree = ET.parse(annotation_path)
            for frame in tree.getroot().findall('frame'):

                # Check that imagepaths match
                image_name = 'img%05d.jpg' % int(frame.attrib['num'])
                image_path = op.join(video_dir, image_name)
                if not image_path in image_paths:
                    raise Exception(
                        'Imagepath %s the xml %s is not a jpeg in dir: %s' %
                        (image_path, annotation_path, video_dir))

                if args.with_display:
                    imagefile = op.relpath(image_path, args.rootdir)
                    img = imreader.imread(imagefile)

                for target in frame.find('target_list').findall('target'):

                    # Parse bbox.
                    box = target.find('box')
                    x1 = int(float(box.attrib['left']))
                    y1 = int(float(box.attrib['top']))
                    width = int(float(box.attrib['width']))
                    height = int(float(box.attrib['height']))
                    logging.debug('Parsed box: %s' %
                                  str([x1, y1, width, height]))
                    c.execute(
                        'INSERT INTO objects(imagefile,x1,y1,width,height) '
                        'VALUES (?,?,?,?,?)',
                        (imagefile, x1, y1, width, height))
                    objectid = c.lastrowid

                    # Parse matches.
                    match = int(target.attrib['id'])
                    match += matches_offset
                    c.execute(
                        'INSERT INTO matches(objectid,match) VALUES (?,?)',
                        (objectid, match))

                    # Parse properties.
                    attribute = target.find('attribute')
                    orientation = float(attribute.attrib['orientation'])
                    speed = float(attribute.attrib['speed'])
                    trajectory_length = int(
                        attribute.attrib['trajectory_length'])
                    vehicle_type = attribute.attrib['vehicle_type']
                    c.execute(
                        'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                        (objectid, 'orientation', str(orientation)))
                    c.execute(
                        'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                        (objectid, 'speed', str(speed)))
                    c.execute(
                        'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                        (objectid, 'vehicle_type', vehicle_type))
                    c.execute(
                        'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                        (objectid, 'trajectory_length',
                         str(trajectory_length)))

                    if args.with_display:
                        util.drawScoredRoi(
                            img, utilBoxes.bbox2roi([x1, y1, width, height]),
                            vehicle_type)

                # Maybe display.
                if args.with_display:
                    cv2.imshow('importDetrac', img[:, :, ::-1])
                    if cv2.waitKey(-1) == 27:
                        args.with_display = False
                        cv2.destroyWindow('importDetrac')
