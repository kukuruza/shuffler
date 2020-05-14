import os, os.path as op
import shutil
import logging
import cv2
import numpy as np
from pprint import pformat
from progressbar import progressbar
import simplejson as json
import matplotlib.pyplot as plt  # for colormaps

from lib.backend.backendDb import objectField

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 28
FONT_SIZE = 0.7
THICKNESS = 3


def maybeDecode(string):
    '''
    Db entries may not be in Unicode. Call this function to use db entries
    to name files in a filesystem.
    '''
    if isinstance(string, bytes):
        return string.decode('UTF-8')
    else:
        return string


def copyWithBackup(in_path, out_path):
    ''' Copy in_path into out_path, which is backed up if already exists. '''

    if not op.exists(in_path):
        raise FileNotFoundError('File does not exist: "%s"' % in_path)
    if op.exists(out_path):
        logging.warning('Will back up existing out_path "%s"' % out_path)
        ext = op.splitext(out_path)[1]
        backup_path = op.splitext(out_path)[0] + '.backup%s' % ext
        if in_path != out_path:
            if op.exists(backup_path):
                os.remove(backup_path)
            os.rename(out_path, backup_path)
        else:
            shutil.copyfile(in_path, backup_path)
    if in_path != out_path:
        # copy input file into the output one
        shutil.copyfile(in_path, out_path)


def bbox2roi(bbox):
    '''
    Args:     [x1, y1, width, height]
    Returns:  [y1, x1, y2, x2]
    '''
    if not (isinstance(bbox, (list, tuple))):
        raise TypeError('Need a list of a tuple, got %s' % type(bbox))
    if not len(bbox) == 4:
        raise ValueError('Need 4 numbers, not %d.' % len(bbox))
    for x in bbox:
        if not (isinstance(x, (int, float))):
            raise TypeError('Each element must be a number, got %s' % type(x))
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError('Bbox %s has negative width or height.' % str(bbox))
    return [bbox[1], bbox[0], bbox[3] + bbox[1], bbox[2] + bbox[0]]


def roi2bbox(roi):
    '''
    Args:     [y1, x1, y2, x2]
    Returns:  [x1, y1, width, height]
    '''
    if not (isinstance(roi, list) or isinstance(roi, tuple)):
        raise TypeError('Need a list of a tuple, got %s' % type(roi))
    if not len(roi) == 4:
        raise ValueError('Need 4 numbers, not %d.' % len(roi))
    for x in roi:
        if not (isinstance(x, (int, float))):
            raise TypeError('Each element must be a number, got %s' % type(x))
    if roi[2] < roi[0] or roi[3] < roi[1]:
        raise ValueError('Roi %s has negative width or height.' % str(roi))
    return [roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0]]


def overlapRatio(roi1, roi2):
    assert (len(roi1) == 4 and len(roi2) == 4)
    if roi1 == roi2: return 1  # same object
    dy = min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])
    dx = min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])
    if dy <= 0 or dx <= 0: return 0
    area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
    area2 = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1])
    inters = dx * dy
    union = area1 + area2 - inters
    logging.debug('inters: %s, union: %s', str(inters), str(union))
    assert (union >= inters and inters > 0)
    return float(inters) / union


def drawScoredPolygon(img, polygon, label=None, score=None):
    '''
    Args:
      img:            Numpy image where to draw the polygon on
      polygon_entry:  A list of tuples (x,y)
      label:          string, name of object
      score:          float, score of object in range [0, 1]
    Returns:
      Nothing, img is changed in-place
    '''
    if label is None: label = ''
    if isinstance(label, bytes):
        label = label.decode('UTF-8')
    if score is None:
        score = 1
    color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
    for i1 in range(len(polygon)):
        i2 = (i1 + 1) % len(polygon)
        cv2.line(img, tuple(polygon[i1]), tuple(polygon[i2]), color, THICKNESS)
    # Draw a label.
    xmin = polygon[0][0]
    ymin = polygon[0][1]
    for i in range(len(polygon) - 1):
        xmin = min(xmin, polygon[i][0])
        ymin = min(ymin, polygon[i][1])
    text_coords = (xmin, ymin - SCALE)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, (0, 0, 0), THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, (255, 255, 255),
                THICKNESS - 1)


def drawScoredRoi(img, roi, label=None, score=None):
    assert score is None or score >= 0 and score <= 1
    if label is None: label = ''
    if isinstance(label, bytes):
        label = label.decode('UTF-8')
    if score is None:
        score = 1
    color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
    cv2.rectangle(img, (roi[1], roi[0]), (roi[3], roi[2]), color, THICKNESS)
    text_coords = (roi[1], roi[0] - SCALE)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, (0, 0, 0), THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, (255, 255, 255),
                THICKNESS - 1)


def drawTextOnImage(img, text):
    '''
    Draw text on image in the corner of the image.
    Returns:
      Nothing. Input "img" is changed in place.
    '''
    imheight, imwidth = img.shape[0:2]
    fontscale = float(imheight) / 500
    thickness = max(imheight // 100, 1)
    offsety = int(fontscale * 40)
    logging.debug('Using offset=%s, fontscale=%s, thickness=%s' %
                  (offsety, fontscale, thickness))
    text_coords = (5, offsety)
    if thickness == 1:
        cv2.putText(img, text, text_coords, FONT, fontscale, (255, 255, 255),
                    thickness)
    else:
        cv2.putText(img, text, text_coords, FONT, fontscale, (0, 0, 0),
                    thickness)
        cv2.putText(img, text, text_coords, FONT, fontscale, (255, 255, 255),
                    thickness - 1)


def applyLabelMappingToMask(mask, labelmap):
    '''
    Args:
      mask:     numpy 2-dim array, the same HxW as img.
      labelmap: dict from mask values to output values Key->Value, where:
                Key is scalar int in [0, 255] or a str type ">=N", "<=N", or "[M,N]".
                If Value is a scalar, the mask stays grayscale,
                if Value is a tuple (r,g,b), the mask is transformed to color.
                Examples {1: 255, 2: 128} or {1: (255,255,255), 2: (128,255,0)}.
                If not specified, the mask is used as is.
    Returns:
      mask      Mapped mask of type float
    '''
    logging.debug('Before mapping, mask had values %s' %
                  set(mask.flatten().tolist()))

    if len(mask.shape) == 3:
        raise NotImplementedError('Color masks are not supported now.')

    logging.debug('Labelmap: %s' % labelmap)
    if len(labelmap) is 0:
        raise ValueError('"labelmap" is empty.')

    dmask = None
    for key, value in labelmap.items():

        # Lazy initialization of dmask.
        if dmask is None:
            if isinstance(value, (list, tuple)) and len(value) == 3:
                # Create a dmask of shape (3, H, W)
                dmask = np.empty(
                    (3, mask.shape[0], mask.shape[1]), dtype=np.uint8) * np.nan
            elif isinstance(value, int):
                # Create a dmask of shape (H, W)
                dmask = np.empty(mask.shape[0:2], dtype=np.uint8) * np.nan
            else:
                raise TypeError('Values of "labelmap" are neither '
                                'a collection of length 3, not a number.')

        if isinstance(key, int):
            roi = mask == key
        elif isinstance(key, str) and len(key) >= 2 and key[0] == '>':
            roi = mask > int(key[1:])
        elif isinstance(key, str) and len(key) >= 2 and key[0] == '<':
            roi = mask < int(key[1:])
        elif isinstance(key, str) and len(key) >= 3 and key[0:2] == '>=':
            roi = mask >= int(key[2:])
        elif isinstance(key, str) and len(key) >= 3 and key[0:2] == '<=':
            roi = mask <= int(key[2:])
        elif isinstance(key, str) and len(
                key) >= 5 and key[0] == '[' and key[-1] == ']' and ',' in key:
            key1, key2 = tuple([int(x) for x in key[1:-1].split(',')])
            roi = np.bitwise_and(mask >= key1, mask <= key2)
        else:
            raise TypeError(
                'Could not interpret the key "%s". Must be int '
                'or string of type "<N", "<=N", ">N", ">=N", "[M,N]".' % key)

        # For grayscale target.
        if len(dmask.shape) == 2 and isinstance(value, int):
            dmask[roi] = value
        # For color target.
        elif len(dmask.shape) == 3 and isinstance(
                value, (list, tuple)) and len(value) == 3:
            dmask[0][roi] = value[0]
            dmask[1][roi] = value[1]
            dmask[2][roi] = value[2]
        else:
            raise ValueError(
                '"labelmap" value %s mismatches dmask\'s shape %s' %
                (value, dmask.shape))
        logging.debug('key: %s, value: %s, numpixels: %d.' %
                      (key, value, np.count_nonzero(roi)))

    if len(dmask.shape) == 3:
        # From (3, H, W) to (H, W, 3)
        dmask = np.transpose(dmask, (1, 2, 0))
        logging.debug('Left %d pixels unmapped (NaN).' %
                      np.count_nonzero(np.isnan(dmask[:, :, 0])))
    else:
        logging.debug('Left %d pixels unmapped (NaN).' %
                      np.count_nonzero(np.isnan(dmask)))

    return dmask


def drawMaskOnImage(img, mask, alpha=0.5, labelmap=None):
    '''
    Draw a mask on the image, with colors.
    Args:
      img:      numpy array
      mask:     numpy 2-dim array, the same HxW as img.
      labelmap: same as arguments of applyLabelMappingToMask
      alpha:    float, alpha=0 means mask is not visible,
                       alpha=1 means img is not visible.
    Returns:
      Output image.
    '''
    if not len(img.shape) == 3:
        raise NotImplementedError('Only color images are supported now.')

    if labelmap is not None:
        mask = applyLabelMappingToMask(mask, labelmap).astype(np.uint8)

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if img.shape[:2] != mask.shape[:2]:
        logging.warning('Image shape %s mismatches mask shape %s.' %
                        (img.shape, mask.shape))
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          cv2.INTER_NEAREST)
    assert img.shape == mask.shape, (img.shape, mask.shape)

    # Overlay mask.
    logging.debug('alpha %.3f' % alpha)
    img = cv2.addWeighted(src1=img,
                          alpha=1 - alpha,
                          src2=mask,
                          beta=alpha,
                          gamma=0)
    return img


def drawMaskAside(img, mask, labelmap=None):
    '''
    Draw a mask on the image, with colors.
    Args:
      img:      numpy array
      mask:     numpy array, the same HxW as img.
      labelmap: see arguments for applyLabelMappingToMask.
    Returns:
      Output image.
    '''
    logging.debug('Image shape: %s, image dtype: %s' % (img.shape, img.dtype))
    logging.debug('Mask shape: %s, mask dtype: %s' % (mask.shape, mask.dtype))
    if labelmap is not None:
        mask = applyLabelMappingToMask(mask, labelmap)

    if len(mask.shape) == 3 and mask.shape[2] == 4 and len(
            img.shape) == 3 and img.shape[2] == 3:
        logging.debug(
            'Mask has alpha channel, but image does not. Add it to image.')
        img = np.concatenate([
            img,
            np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        ],
                             axis=2)
    if len(mask.shape) == 2 and len(img.shape) == 3 and img.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if mask.dtype == np.uint16:  # 16 bit is transferred to 8 bit.
        mask = (mask // 256).astype(np.uint8)
    if img.shape[:2] != mask.shape[:2]:
        logging.warning('Image shape %s mismatches mask shape %s.' %
                        (img.shape, mask.shape))
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          cv2.INTER_NEAREST)
    assert img.shape == mask.shape, (img.shape, mask.shape)

    # Draw mask aside.
    img = np.hstack([img, mask])
    return img


def bboxes2polygons(cursor, objectid):
    # If there are already polygon entries, do nothing.
    cursor.execute('SELECT COUNT(1) FROM polygons WHERE objectid=?',
                   (objectid, ))
    if cursor.fetchone()[0] > 0:
        return

    cursor.execute('SELECT * FROM objects WHERE objectid=?', (objectid, ))
    object_entry = cursor.fetchone()
    if object_entry is None:
        raise ValueError('Objectid %d does not exist.' % objectid)
    y1, x1, y2, x2 = objectField(object_entry, 'roi')
    for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        cursor.execute(
            'INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,"bounding box")',
            (objectid, x, y))
    logging.debug(
        'Added polygon from bbox y1=%d,x1=%d,y2=%d,x2=%d to objectid %d' %
        (y1, x1, y2, x2, objectid))


def polygons2bboxes(cursor, objectid):
    cursor.execute('SELECT COUNT(1) FROM polygons WHERE objectid=?',
                   (objectid, ))
    if cursor.fetchone()[0] == 0:
        logging.debug('Objectid %d does not have polygons.' % objectid)
        return
    cursor.execute(
        'UPDATE objects SET x1=(SELECT MIN(x) FROM polygons '
        'WHERE objectid=?) WHERE objectid=?', (objectid, objectid))
    cursor.execute(
        'UPDATE objects SET y1=(SELECT MIN(y) FROM polygons '
        'WHERE objectid=?) WHERE objectid=?', (objectid, objectid))
    cursor.execute(
        'UPDATE objects SET width=(SELECT MAX(x) FROM polygons '
        'WHERE objectid=?) - x1 WHERE objectid=?', (objectid, objectid))
    cursor.execute(
        'UPDATE objects SET height=(SELECT MAX(y) FROM polygons '
        'WHERE objectid=?) - y1 WHERE objectid=?', (objectid, objectid))


def polygons2mask(cursor, objectid):

    cursor.execute(
        'SELECT i.width,i.height FROM images i INNER JOIN '
        'objects o ON i.imagefile=o.imagefile '
        'WHERE objectid=?', (objectid, ))
    width, height = cursor.fetchone()
    mask = np.zeros((height, width), dtype=np.int32)

    # Iterate multiple polygons (if any) of the object.
    cursor.execute('SELECT DISTINCT(name) FROM polygons WHERE objectid=?',
                   (objectid, ))
    polygon_names = cursor.fetchall()
    logging.debug('objectid %d has %d.' % (objectid, len(polygon_names)))
    for polygon_name, in polygon_names:

        # Draw a polygon.
        if polygon_name is None:
            cursor.execute('SELECT x,y FROM polygons WHERE objectid=?',
                           (objectid, ))
        else:
            cursor.execute(
                'SELECT x,y FROM polygons WHERE objectid=? AND name=?',
                (objectid, polygon_name))
        pts = [[pt[0], pt[1]] for pt in cursor.fetchall()]
        logging.debug('Polygon "%s" of object %d consists of points: %s' %
                      (polygon_name, objectid, str(pts)))
        if len(pts) < 3:
            logging.debug('Skipping polygon without enough points')
            continue
        cv2.fillPoly(mask, np.asarray([pts], dtype=np.int32), 255)

    mask = mask.astype(np.uint8)
    return mask