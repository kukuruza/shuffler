import os, sys, os.path as op
import numpy as np
import cv2
import collections
import logging
from glob import glob
import shutil
import sqlite3
from progressbar import progressbar
from pprint import pformat

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.utils import util


def add_parsers(subparsers):
    importKittiParser(subparsers)


# Data Format Description
# =======================

# The data for training and testing can be found in the corresponding folders.
# The sub-folders are structured as follows:

#   - image_02/ contains the left color camera images (png)
#   - label_02/ contains the left color camera label files (plain text files)
#   - calib/ contains the calibration for all four cameras (plain text file)

# The label files contain the following information, which can be read and
# written using the matlab tools (readLabels.m, writeLabels.m) provided within
# this devkit. All values (numerical or strings) are separated via spaces,
# each row corresponds to one object. The 15 columns represent:

# #Values    Name      Description
# ----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.

# Here, 'DontCare' labels denote regions in which objects have not been labeled,
# for example because they have been too far away from the laser scanner. To
# prevent such objects from being counted as false positives our evaluation
# script will ignore objects detected in don't care regions of the test set.
# You can use the don't care labels in the training set to avoid that your object
# detector is harvesting hard negatives from those areas, in case you consider
# non-object regions from the training images as negative examples.

# The coordinates in the camera coordinate system can be projected in the image
# by using the 3x4 projection matrix in the calib folder, where for the left
# color camera for which the images are provided, P2 must be used. The
# difference between rotation_y and alpha is, that rotation_y is directly
# given in camera coordinates, while alpha also considers the vector from the
# camera center to the object center, to compute the relative orientation of
# the object with respect to the camera. For example, a car which is facing
# along the X-axis of the camera coordinate system corresponds to rotation_y=0,
# no matter where it is located in the X/Z plane (bird's eye view), while
# alpha is zero only, when this object is located along the Z-axis of the
# camera. When moving the car away from the Z-axis, the observation angle
# will change.

# To project a point from Velodyne coordinates into the left color image,
# you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
# For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

# Note: All matrices are stored row-major, i.e., the first values correspond
# to the first row. R0_rect contains a 3x3 matrix which you need to extend to
# a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
# Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
# in the same way!

# Note, that while all this information is available for the training data,
# only the data which is actually needed for the particular benchmark must
# be provided to the evaluation server. However, all 15 values must be provided
# at all times, with the unused ones set to their default values (=invalid) as
# specified in writeLabels.m. Additionally a 16'th value must be provided
# with a floating value of the score for a particular detection, where higher
# indicates higher confidence in the detection. The range of your scores will
# be automatically determined by our evaluation server, you don't have to
# normalize it, but it should be roughly linear. If you use writeLabels.m for
# writing your results, this function will take care of storing all required
# data correctly.


def _parseObject(c, line, imagefile):
    words = line.split(' ')
    # 1    type         Describes the type of object: 'Car', 'Van', 'Truck',
    #                   'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    #                   'Misc' or 'DontCare'
    name = words[0]
    # 1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
    #                   truncated refers to the object leaving image boundaries
    truncated = words[1]
    # 1    occluded     Integer (0,1,2,3) indicating occlusion state:
    #                   0 = fully visible, 1 = partly occluded
    #                   2 = largely occluded, 3 = unknown
    occluded = words[2]
    # 1    alpha        Observation angle of object, ranging [-pi..pi]
    alpha = words[3]
    # 4    bbox         2D bounding box of object in the image (0-based index):
    #                   contains left, top, right, bottom pixel coordinates
    x1 = int(float(words[4]))
    y1 = int(float(words[5]))
    width = int(float(words[6]) - float(words[4]) + 1)
    height = int(float(words[7]) - float(words[5]) + 1)
    # 3    dimensions   3D object dimensions: height, width, length (in meters)
    dim_height = words[8]
    dim_width = words[9]
    dim_length = words[10]
    # 3    location     3D object location x,y,z in camera coordinates (in meters)
    loc_x = words[11]
    loc_y = words[12]
    loc_z = words[13]
    # 1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    rotation_y = words[14]
    # 1    score        Only for results: Float, indicating confidence in
    #                   detection, needed for p/r curves, higher is better.
    score = float(words[15]) if len(words) == 16 else None
    c.execute(
        'INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
        'VALUES (?,?,?,?,?,?,?)',
        (imagefile, x1, y1, width, height, name, score))
    objectid = c.lastrowid
    s = 'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)'
    c.execute(s, (objectid, 'truncated', truncated))
    c.execute(s, (objectid, 'occluded', occluded))
    c.execute(s, (objectid, 'alpha', alpha))
    c.execute(s, (objectid, 'dim_height', dim_height))
    c.execute(s, (objectid, 'dim_width', dim_width))
    c.execute(s, (objectid, 'dim_length', dim_length))
    c.execute(s, (objectid, 'loc_x', loc_x))
    c.execute(s, (objectid, 'loc_y', loc_y))
    c.execute(s, (objectid, 'loc_z', loc_z))
    c.execute(s, (objectid, 'rotation_y', rotation_y))
    return objectid


def importKittiParser(subparsers):
    parser = subparsers.add_parser(
        'importKitti',
        description='Import KITTI annotations into a db. Note that the '
        'bounding box is stored as integer, and the decimal part is lost.')
    parser.set_defaults(func=importKitti)
    parser.add_argument('--images_dir',
                        required=True,
                        help='Directory with .png images. '
                        'E.g. "kitti/data_object_image_2/training/image_2".')
    parser.add_argument('--detection_dir',
                        help='Directory with .txt annotations of objects. '
                        'E.g. "kitti/data_object_image_2/training/label_2".')
    parser.add_argument(
        '--segmentation_dir',
        help='Directory with 8bit masks for semantic and 16bit masks '
        'for instance segmentation annotations. '
        'E.g. "kitti/data_object_image_2/training/label_2".')
    parser.add_argument('--with_display', action='store_true')


def importKitti(c, args):
    if args.with_display:
        imreader = backendMedia.MediaReader(args.rootdir)

    image_paths = sorted(glob(op.join(args.images_dir, '*.png')))
    logging.info('Found %d PNG images in %s' %
                 (len(image_paths), args.images_dir))

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

        # Detection annotations.
        if args.detection_dir:
            detection_path = op.join(args.detection_dir, '%s.txt' % filename)
            if not op.exists(detection_path):
                raise FileNotFoundError('Annotation file not found at "%s".' %
                                        detection_path)

            # Read annotation file.
            with open(detection_path) as f:
                lines = f.read().splitlines()
            logging.debug('Read %d lines from detection file "%s"' %
                          (len(lines), detection_path))
            for line in lines:
                objectid = _parseObject(c, line, imagefile)

                if args.with_display:
                    c.execute('SELECT * FROM objects WHERE objectid=?',
                              (objectid, ))
                    object_entry = c.fetchone()
                    name = backendDb.objectField(object_entry, 'name')
                    roi = backendDb.objectField(object_entry, 'roi')
                    score = backendDb.objectField(object_entry, 'score')
                    util.drawScoredRoi(img, roi, name, score=score)

        # Segmentation annotations.
        if args.segmentation_dir:
            segmentation_path = op.join(args.segmentation_dir,
                                        '%s.png' % filename)
            if not op.exists(segmentation_path):
                raise FileNotFoundError('Annotation file not found at "%s".' %
                                        segmentation_path)

            # Add image to the database.
            maskfile = op.relpath(segmentation_path, args.rootdir)
            c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                      (maskfile, imagefile))

            if args.with_display:
                mask = imreader.maskread(maskfile)
                img = util.drawMaskAside(img, mask, labelmap=None)

        # Maybe display.
        if args.with_display:
            cv2.imshow('importKitti', img[:, :, ::-1])
            if cv2.waitKey(-1) == 27:
                args.with_display = False
                cv2.destroyWindow('importKitti')
