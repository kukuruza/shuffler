import os, sys, os.path as op
import numpy as np
import cv2
import logging
from ast import literal_eval
from pprint import pformat
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.utils.util import drawTextOnImage, drawMaskOnImage, drawMaskAside
from lib.utils.util import bbox2roi, drawScoredRoi, drawScoredPolygon
from lib.utils.util import FONT, SCALE, FONT_SIZE, THICKNESS
from lib.backend.backendDb import deleteImage, deleteObject, imageField, objectField, polygonField
from lib.backend.backendMedia import MediaReader, normalizeSeparators


def add_parsers(subparsers):
    displayImagesPltParser(subparsers)


def displayImagesPltParser(subparsers):
    parser = subparsers.add_parser(
        'displayImagesPlt',
        description='Display a set of images in a plot. '
        'Similar to examineImages, but for matplotlib')
    parser.set_defaults(func=displayImagesPlt)
    parser.add_argument('--limit',
                        type=int,
                        default=1,
                        help='How many images to put into a plot.')
    parser.add_argument('--where_image',
                        default='TRUE',
                        help='the SQL "where" clause for the "images" table.')
    parser.add_argument('--mask_mapping_dict',
                        help='A mapping to display values in maskfile. '
                        'E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask_aside',
                       action='store_true',
                       help='Image and mask side by side.')
    group.add_argument('--mask_alpha',
                       type=float,
                       help='Mask will be overlayed on the image.'
                       'Transparency to overlay the label mask with, '
                       '1 means cant see the image behind the mask.')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--with_objects',
                        action='store_true',
                        help='draw all objects on top of the image.')
    parser.add_argument('--with_score',
                        action='store_true',
                        help='draw image score on top of the image.')
    parser.add_argument('--with_imagefile',
                        action='store_true',
                        help='draw imagefile on top of the image.')


def displayImagesPlt(c, args):
    c.execute('SELECT * FROM images WHERE (%s)' % args.where_image)
    image_entries = c.fetchall()
    logging.info('%d images found.' % len(image_entries))
    if len(image_entries) == 0:
        logging.error('There are no images. Exiting.')
        return

    if args.shuffle:
        np.random.shuffle(image_entries)

    if len(image_entries) < args.limit:
        image_entries = image_entries[:args.limit]

    _, axes = plt.subplots(nrows=min(args.limit, len(image_entries)), ncols=1)
    if args.limit == 1:
        axes = [axes]

    imreader = MediaReader(rootdir=args.rootdir)

    # For overlaying masks.
    labelmap = literal_eval(
        args.mask_mapping_dict) if args.mask_mapping_dict else None
    logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

    for image_entry, ax in zip(image_entries, axes):

        imagefile = imageField(image_entry, 'imagefile')
        maskfile = imageField(image_entry, 'maskfile')
        imname = (imageField(image_entry, 'name'))
        imscore = (imageField(image_entry, 'score'))
        logging.info('Imagefile "%s"' % imagefile)
        logging.debug('Image name="%s", score=%s' % (imname, imscore))
        image = imreader.imread(imagefile)

        # Overlay the mask.
        if maskfile is not None:
            mask = imreader.maskread(maskfile)
            if args.mask_aside:
                image = drawMaskAside(image, mask, labelmap=labelmap)
            elif args.mask_alpha is not None:
                image = drawMaskOnImage(image,
                                        mask,
                                        alpha=args.mask_alpha,
                                        labelmap=labelmap)
        else:
            logging.info('No mask for this image.')

        # Put the objects on top of the image.
        if args.with_objects:
            c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
            object_entries = c.fetchall()
            logging.info('Found %d objects for image %s' %
                         (len(object_entries), imagefile))
            for object_entry in object_entries:
                objectid = objectField(object_entry, 'objectid')
                roi = objectField(object_entry, 'roi')
                score = objectField(object_entry, 'score')
                name = objectField(object_entry, 'name')
                logging.info('objectid: %d, roi: %s, score: %s, name: %s' %
                             (objectid, roi, score, name))
                c.execute('SELECT * FROM polygons WHERE objectid=?',
                          (objectid, ))
                polygon_entries = c.fetchall()
                if len(polygon_entries) > 0:
                    logging.info('showing object with a polygon.')
                    polygon = [(int(polygonField(p, 'x')),
                                int(polygonField(p, 'y')))
                               for p in polygon_entries]
                    drawScoredPolygon(image, polygon, label=name, score=score)
                elif roi is not None:
                    logging.info('showing object with a bounding box.')
                    drawScoredRoi(image, roi, label=name, score=score)
                else:
                    logging.warning(
                        'Neither polygon, nor bbox is available for objectid %d'
                        % objectid)

        # Overlay imagefile.
        title = ""
        if args.with_imagefile:
            title += '%s ' % op.basename(normalizeSeparators(imagefile))
        # Overlay score.
        if args.with_score:
            title += '%.3f ' % imscore
        ax.set_title(title)

        # Display
        #ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()