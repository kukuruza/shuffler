import os.path as op
import numpy as np
import logging
from ast import literal_eval
from pprint import pformat
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from lib.utils import util
from lib.backend import backendDb
from lib.backend import backendMedia


def add_parsers(subparsers):
    displayImagesPltParser(subparsers)


def drawScoredRoi(ax, roi, label=None, score=None):
    '''
    Draw a bounding box on top of Matplotlib axes.
    Args:
      ax:       Matplotlib axes.
      roi:      List/tuple [y1, x1, y2, x2].
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
    Return:
      Nothing, 'img' is changed in place.
    '''
    if score is None:
        score = 1.
    cmap = cm.get_cmap('jet').reversed()
    rgba = cmap(float(score))
    rect = patches.Rectangle(xy=(roi[1], roi[0]),
                             width=roi[3] - roi[1],
                             height=roi[2] - roi[0],
                             linewidth=1,
                             edgecolor=rgba,
                             facecolor='none')
    ax.add_patch(rect)

    if label:
        text = ax.text(x=roi[1], y=roi[0] - 5, s=label, c='white')
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])


def drawScoredPolygon(ax, polygon, label=None, score=None):
    '''
    Draw a polygon on top of Matplotlib axes.
    Args:
      ax:       Matplotlib axes.
      polygon:  List of tuples (x,y)
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
    Returns:
      Nothing, 'img' is changed in place.
    '''
    if score is None:
        score = 1.
    cmap = cm.get_cmap('jet').reversed()
    rgba = cmap(float(score))
    polygon = np.array(polygon)
    rect = patches.Polygon(xy=polygon,
                           linewidth=1,
                           edgecolor=rgba,
                           facecolor='none')
    ax.add_patch(rect)

    xmin = polygon[:, 0].min()
    ymin = polygon[:, 1].min()
    if label is not None:
        # if isinstance(label, bytes):
        label = util.maybeDecode(label)
        if label:
            text = ax.text(x=xmin, y=ymin - 5, s=label, c='white')
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()
            ])


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
    parser.add_argument('--ncols',
                        type=int,
                        default=1,
                        help='Number of columns for matplotlib figure.')


def displayImagesPlt(c, args):
    c.execute('SELECT * FROM images WHERE (%s)' % args.where_image)
    image_entries = c.fetchall()
    logging.info('%d images found.', len(image_entries))
    if len(image_entries) == 0:
        logging.error('There are no images. Exiting.')
        return

    if args.shuffle:
        np.random.shuffle(image_entries)

    if len(image_entries) < args.limit:
        image_entries = image_entries[:args.limit]

    def _getNumRows(total, ncols):
        nrows = int((total - 1) / ncols) + 1
        logging.info('Grid: %dx%d from the total of %d', nrows, ncols, total)
        return nrows

    if args.limit < len(image_entries):
        image_entries = image_entries[:args.limit]
    nrows = _getNumRows(len(image_entries), args.ncols)

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For overlaying masks.
    labelmap = literal_eval(
        args.mask_mapping_dict) if args.mask_mapping_dict else None
    logging.info('Parsed mask_mapping_dict to %s', pformat(labelmap))

    for i_image, image_entry in enumerate(image_entries):
        ax = plt.subplot(nrows, args.ncols, i_image + 1)

        imagefile = backendDb.imageField(image_entry, 'imagefile')
        maskfile = backendDb.imageField(image_entry, 'maskfile')
        imname = backendDb.imageField(image_entry, 'name')
        imscore = backendDb.imageField(image_entry, 'score')
        logging.info('Imagefile "%s"', imagefile)
        logging.debug('Image name="%s", score=%s', imname, imscore)
        image = imreader.imread(imagefile)

        # Overlay the mask.
        if maskfile is not None:
            mask = imreader.maskread(maskfile)
            if args.mask_aside:
                image = util.drawMaskAside(image, mask, labelmap=labelmap)
            elif args.mask_alpha is not None:
                image = util.drawMaskOnImage(image,
                                             mask,
                                             alpha=args.mask_alpha,
                                             labelmap=labelmap)
        else:
            logging.info('No mask for this image.')

        # Put the objects on top of the image.
        if args.with_objects:
            c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
            object_entries = c.fetchall()
            logging.info('Found %d objects for image %s', len(object_entries),
                         imagefile)
            for object_entry in object_entries:
                objectid = backendDb.objectField(object_entry, 'objectid')
                roi = backendDb.objectField(object_entry, 'roi')
                score = backendDb.objectField(object_entry, 'score')
                name = backendDb.objectField(object_entry, 'name')
                logging.info('objectid: %d, roi: %s, score: %s, name: %s',
                             objectid, roi, score, name)
                c.execute('SELECT * FROM polygons WHERE objectid=?',
                          (objectid, ))
                polygon_entries = c.fetchall()
                if len(polygon_entries) > 0:
                    logging.info('showing object with a polygon.')
                    polygon = [(int(backendDb.polygonField(p, 'x')),
                                int(backendDb.polygonField(p, 'y')))
                               for p in polygon_entries]
                    drawScoredPolygon(ax, polygon, label=name, score=score)
                elif roi is not None:
                    logging.info('showing object with a bounding box.')
                    drawScoredRoi(ax, roi, label=name, score=score)
                else:
                    logging.warning(
                        'Neither polygon, nor bbox is available for objectid %d',
                        objectid)

        # Overlay imagefile.
        title = ""
        if args.with_imagefile:
            title += '%s ' % op.basename(
                backendMedia.normalizeSeparators(imagefile))
        # Overlay score.
        if args.with_score:
            title += '%.3f ' % imscore if imscore is not None else ''
        ax.set_title(title)

        # Display
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
