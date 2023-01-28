import os, os.path as op
import shutil
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

from shuffler.backend import backend_db
from shuffler.utils import boxes as boxes_utils

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 28
FONT_SIZE = 1.2
THICKNESS = 2
TEXT_COLOR = (0, 0, 0)
TEXT_BACKCOLOR = (255, 255, 255)


def validateFileName(filename):
    invalid = '<>:\'"/\\|?+=!@#$%^&*,.~`'
    filename = maybeDecode(filename)
    for char in invalid:
        filename = filename.replace(char, '_')
    return filename


def maybeDecode(string):
    '''
    Db entries may be not in Unicode. Call this function to use db entries
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
        logging.warning('Will back up existing out_path "%s"', out_path)
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


def drawScoredRoi(img, roi, label=None, score=None):
    '''
    Draw a bounding box on top of image.
    Args:
      img:      Numpy color image.
      roi:      List/tuple [y1, x1, y2, x2].
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
    Return:
      Nothing, 'img' is changed in place.
    '''
    assert len(img.shape) == 3 and img.shape[2] == 3, (
        'Image must be color, but is grayscale: %s' % str(img.shape))
    assert score is None or score >= 0 and score <= 1
    if label is None:
        label = ''
    label = maybeDecode(label)
    if score is None:
        score = 1
    color = tuple([int(x * 255)
                   for x in plt.cm.afmhot_r(float(score))][0:3])[::-1]
    roi = [int(a) for a in roi]
    cv2.rectangle(img, (roi[1], roi[0]), (roi[3], roi[2]), color, THICKNESS)
    text_coords = (roi[1], roi[0] - SCALE)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_COLOR,
                THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_BACKCOLOR,
                THICKNESS - 1)


def drawScoredPolygon(img, polygon, label=None, score=None):
    '''
    Args:
      img:      Numpy color image.
      polygon:  List of tuples (x,y)
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
    Returns:
      Nothing, 'img' is changed in place.
    '''
    assert len(img.shape) == 3 and img.shape[2] == 3, (
        'Image must be color, but is grayscale: %s' % str(img.shape))
    if label is None:
        label = ''
    label = maybeDecode(label)
    if score is None:
        score = 1
    color = tuple([int(x * 255)
                   for x in plt.cm.afmhot_r(float(score))][0:3])[::-1]
    polygon = [(int(x), int(y)) for x, y in polygon]
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
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_COLOR,
                THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_BACKCOLOR,
                THICKNESS - 1)


def drawTextOnImage(img, text):
    '''
    Draw text on image in the corner of the image.
    Returns:
      Nothing. Input "img" is changed in place.
    '''
    imheight, _ = img.shape[0:2]
    fontscale = float(imheight) / 500
    thickness = max(imheight // 100, 1)
    offsety = int(fontscale * 40)
    logging.debug('Using offset=%s, fontscale=%s, thickness=%s', offsety,
                  fontscale, thickness)
    text_coords = (5, offsety)
    if thickness == 1:
        cv2.putText(img, text, text_coords, FONT, fontscale, TEXT_BACKCOLOR,
                    thickness)
    else:
        cv2.putText(img, text, text_coords, FONT, fontscale, TEXT_COLOR,
                    thickness)
        cv2.putText(img, text, text_coords, FONT, fontscale, TEXT_BACKCOLOR,
                    thickness - 1)


def applyMaskMapping(mask, labelmap):
    '''
    Applies the provided mapping to each of mask pixels.
    Args:
      mask:     numpy 2-dim array. Color masks are not supported now.
      labelmap: a dict for remapping mask values.
                Keys can be a combination of:
                - Scalar int in range [0, 255].
                - Strings ">N", "<N", ">=N", "<=N", or "[M,N]". Treated as a range.
                - If there is a map pixel not in the map, it is left unchanged.
                Values can be:
                - Scalars. Then the mask stays grayscale,
                - Tuple (r,g,b). Then the mask is transformed to color.

                Examples:
                - {1: 255, 2: 128}
                        remaps 1 to 255, 2 to 128.
                - {1: (255,255,255), 2: (128,255,0)}
                        remaps 1 to (255,255,255) and 2 to (128,255,0)
                - {"[0, 254]": 0, 255: 1}
                        remaps any pixel in range [0, 254] to 0 and 255 to 1.
    Returns:
      mask      Mapped mask of type float
    '''
    logging.debug('Before mapping, mask had values %s',
                  set(mask.flatten().tolist()))

    if len(mask.shape) == 3:
        raise NotImplementedError('Color masks are not supported now.')

    logging.debug('Labelmap: %s', labelmap)
    if len(labelmap) == 0:
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
        logging.debug('key: %s, value: %s, numpixels: %d.', key, value,
                      np.count_nonzero(roi))

    if len(dmask.shape) == 3:
        # From (3, H, W) to (H, W, 3)
        dmask = np.transpose(dmask, (1, 2, 0))
        logging.debug('Left %d pixels unmapped (NaN).',
                      np.count_nonzero(np.isnan(dmask[:, :, 0])))
    else:
        logging.debug('Left %d pixels unmapped (NaN).',
                      np.count_nonzero(np.isnan(dmask)))

    return dmask


def drawMaskOnImage(img, mask, alpha=0.5, labelmap=None):
    '''
    Draw a mask on the image, with colors.
    Args:
      img:      numpy array
      mask:     numpy 2-dim array, the same HxW as img.
      labelmap: same as arguments of applyMaskMapping
      alpha:    float, alpha=0 means mask is not visible,
                       alpha=1 means img is not visible.
    Returns:
      Output image.
    '''
    if not len(img.shape) == 3:
        raise NotImplementedError('Only color images are supported now.')

    if labelmap is not None:
        mask = applyMaskMapping(mask, labelmap).astype(np.uint8)

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    if img.shape[:2] != mask.shape[:2]:
        logging.warning('Image shape %s mismatches mask shape %s.', img.shape,
                        mask.shape)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          cv2.INTER_NEAREST)
    assert img.shape == mask.shape, (img.shape, mask.shape)

    # Overlay mask.
    logging.debug('alpha %.3f', alpha)
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
      labelmap: see arguments for applyMaskMapping.
    Returns:
      Output image.
    '''
    logging.debug('Image shape: %s, image dtype: %s', img.shape, img.dtype)
    logging.debug('Mask shape: %s, mask dtype: %s', mask.shape, mask.dtype)
    if labelmap is not None:
        mask = applyMaskMapping(mask, labelmap)

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
        logging.warning('Image shape %s mismatches mask shape %s.', img.shape,
                        mask.shape)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          cv2.INTER_NEAREST)
    assert img.shape == mask.shape, (img.shape, mask.shape)

    # Draw mask aside.
    img = np.hstack([img, mask])
    return img


def bboxes2polygons(cursor, objectid):
    ''' A rectangular polygon is added to objects that are missing polygons. '''

    # If there are already polygon entries, do nothing.
    cursor.execute('SELECT COUNT(1) FROM polygons WHERE objectid=?',
                   (objectid, ))
    if cursor.fetchone()[0] > 0:
        return

    cursor.execute('SELECT * FROM objects WHERE objectid=?', (objectid, ))
    object_entry = cursor.fetchone()
    if object_entry is None:
        raise ValueError('Objectid %d does not exist.' % objectid)
    y1, x1, y2, x2 = backend_db.objectField(object_entry, 'roi')
    for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        cursor.execute(
            'INSERT INTO polygons(objectid,x,y,name) VALUES (?,?,?,"bounding box")',
            (objectid, x, y))
    logging.debug(
        'Added polygon from bbox y1=%f,x1=%f,y2=%f,x2=%f to objectid %d', y1,
        x1, y2, x2, objectid)


def polygons2bboxes(cursor, objectid):
    '''
    A bounding box is created around objects that don't have it
    via enclosing the polygon. The polygon is assumed to be present.
    '''
    cursor.execute(
        'SELECT COUNT(1) FROM polygons WHERE objectid=? GROUP BY name',
        (objectid, ))
    num_distinct_polygons = len(cursor.fetchall())
    if num_distinct_polygons == 0:
        logging.debug('Objectid %d does not have polygons.', objectid)
        return
    if num_distinct_polygons > 1:
        raise ValueError(
            'Object %d has %d polygons (polygons with different names). '
            'Merging them is not supported.' %
            (objectid, num_distinct_polygons))

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
    ''' A mask is created around each object by painting inside polygons. '''

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
    logging.debug('objectid %d has %d.', objectid, len(polygon_names))
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
        logging.debug('Polygon "%s" of object %d consists of points: %s',
                      polygon_name, objectid, str(pts))
        if len(pts) < 3:
            logging.debug('Skipping polygon without enough points')
            continue
        cv2.fillPoly(mask, np.asarray([pts], dtype=np.int32), 255)

    mask = mask.astype(np.uint8)
    return mask


def getIntersectingObjects(objects1, objects2, IoU_threshold, same_id_ok=True):
    '''
    Given two lists of objects find pairs that intersect by IoU_threshold.
    Objects are assumed to be in the same image.

    Args:
      objects1, objects2:  A list of object entries.
                           Each entry is the whole row in the 'objects' table.
      IoU_threshold:       A float in range [0, 1].
    Returns:
      A list of tuples. Each tuple has objectid of an entry in 'objects1' and
                           objectid of an entry in 'objects2'.
    '''

    # Compute pairwise distances between rectangles.
    # TODO: possibly can optimize in future to avoid O(N^2) complexity.
    pairwise_IoU = np.zeros(shape=(len(objects1), len(objects2)), dtype=float)
    for i1, object1 in enumerate(objects1):
        for i2, object2 in enumerate(objects2):
            # Do not merge an object with itself.
            objectid1 = backend_db.objectField(object1, 'objectid')
            objectid2 = backend_db.objectField(object2, 'objectid')
            if objectid1 == objectid2 and not same_id_ok:
                pairwise_IoU[i1, i2] = np.nan
            else:
                roi1 = backend_db.objectField(object1, 'roi')
                roi2 = backend_db.objectField(object2, 'roi')
                pairwise_IoU[i1, i2] = boxes_utils.getIoU(roi1, roi2)
    logging.debug('getIntersectingObjects got Pairwise_IoU:\n%s',
                  np.array2string(pairwise_IoU, precision=1))

    # Greedy search for pairs.
    pairs_to_merge = []
    for _ in range(min(len(objects1), len(objects2))):
        i1, i2 = np.unravel_index(np.argmax(pairwise_IoU), pairwise_IoU.shape)
        IoU = pairwise_IoU[i1, i2]
        logging.debug('Next object at indices [%d, %d]. IoU: %s', i1, i2,
                      str(IoU))
        # Stop if no more good pairs.
        if np.isnan(IoU) or IoU < IoU_threshold:
            break
        # Disable these objects for the next step.
        pairwise_IoU[i1, :] = 0.
        pairwise_IoU[:, i2] = 0.
        # Add a pair to the list.
        objectid1 = backend_db.objectField(objects1[i1], 'objectid')
        objectid2 = backend_db.objectField(objects2[i2], 'objectid')
        pairs_to_merge.append((objectid1, objectid2))
        name1 = backend_db.objectField(objects1[i1], 'name')
        name2 = backend_db.objectField(objects2[i2], 'name')
        logging.debug('Matching objects %d (%s) and %d (%s) with IoU %f.',
                      objectid1, name1, objectid2, name2, IoU)

    return pairs_to_merge


def makeExportedImageName(tgt_dir,
                          imagefile,
                          dirtree_level_for_name=1,
                          fix_invalid_image_names=True):
    # Keep dirtree_level_for_name - 1 directories in tgt_name.
    # Replace the directory delimeter with "_".
    tgt_name = imagefile[::-1]
    tgt_name = tgt_name.replace(op.sep, "_", dirtree_level_for_name - 1)
    tgt_name = tgt_name[::-1]
    tgt_name = op.basename(tgt_name)

    if fix_invalid_image_names:
        tgt_name_fixed = re.sub(r'[^a-zA-Z0-9_.-]', '_', tgt_name)
        if tgt_name_fixed != tgt_name:
            logging.info(
                'Replaced invalid characters in image name "%s" to "%s"',
                tgt_name, tgt_name_fixed)
            tgt_name = tgt_name_fixed

    tgt_path = op.join(tgt_dir, tgt_name)

    logging.debug('The target path is %s', tgt_path)
    if op.exists(tgt_path):
        message = 'File with this name %s already exists. ' % tgt_path
        if dirtree_level_for_name == 1:
            message += 'There may be images with the same names in '
            'several dirs. Use --full_imagefile_as_name. '
        raise FileExistsError(message)
    return tgt_path


def getMatchPolygons(polygons1, polygons2, threshold, ignore_name=True):
    '''
    Given two lists of polygons points find pairs that are within threshold.
    Polygons are assumed to belong to the same object.

    Args:
      polygons1, polygons2: A list of polygon entries.
                            Each entry is the whole row in the 'polygon' table.
      threshold:            (float) Anything above it is not matched.
      ignore_name:          (bool) Whether to match points with different names.
    Returns:
      pairs_to_merge:       A list of tuples. Each tuple has [polygon]id of an
                            entry in 'polygons1' and [polygon]id of an entry in
                            'polygons2'.
    '''
    # Compute pairwise distances between rectangles.
    # TODO: possibly can optimize in future to avoid O(N^2) complexity.
    pairwise_dist = np.zeros(shape=(len(polygons1), len(polygons2)),
                             dtype=float)
    for i1, polygon_entry1 in enumerate(polygons1):
        for i2, polygon_entry2 in enumerate(polygons2):
            x1 = backend_db.polygonField(polygon_entry1, 'x')
            y1 = backend_db.polygonField(polygon_entry1, 'y')
            name1 = backend_db.polygonField(polygon_entry1, 'name')
            x2 = backend_db.polygonField(polygon_entry2, 'x')
            y2 = backend_db.polygonField(polygon_entry2, 'y')
            name2 = backend_db.polygonField(polygon_entry2, 'name')
            pairwise_dist[i1, i2] = np.linalg.norm(
                np.array([x1, y1], dtype=float) -
                np.array([x2, y2], dtype=float)) + (
                    0 if ignore_name or name1 == name2 else np.inf)
    logging.debug('getMatchPolygons got pairwise_dist:\n%s',
                  np.array2string(pairwise_dist, precision=1))

    # Greedy search for pairs.
    pairs_to_merge = []
    for _ in range(min(len(polygons1), len(polygons2))):
        i1, i2 = np.unravel_index(np.argmin(pairwise_dist),
                                  pairwise_dist.shape)
        dist = pairwise_dist[i1, i2]
        # Stop if no more good pairs.
        if dist > threshold:
            logging.debug('Already got a non-matching pair: %d and %d', i1, i2)
            break
        # If there are multiple matches under the threshold, it's a problem
        # because matching should be non-ambiguous. In so, raise an error.
        pairwise_dist[i1, i2] = np.inf
        if np.min(pairwise_dist[i1, :]) < threshold:
            i2_ambiguous = np.argmin(pairwise_dist[i1, :].flatten())
            raise ValueError('Polygon points id=%d can be matched with two '
                             'points id=%d and id=%d, which is ambiguous' %
                             (i1, i2, i2_ambiguous))
        if np.min(pairwise_dist[:, i2]) < threshold:
            i1_ambiguous = np.argmin(pairwise_dist[:, i2].flatten())
            raise ValueError('Two polygon points id=%d and id=%d can be '
                             'matched with id=%d, which is ambiguous' %
                             (i1, i1_ambiguous, i2))
        # np.min(pairwise_dist[i1, :])
        # Disable these polygons for the next step.
        pairwise_dist[i1, :] = np.inf
        pairwise_dist[:, i2] = np.inf
        # Add a pair to the list.
        id1 = backend_db.polygonField(polygons1[i1], 'id')
        id2 = backend_db.polygonField(polygons2[i2], 'id')
        pairs_to_merge.append((id1, id2))
        logging.debug(
            'Matched points %d and %d (indices %d and %d) with distance %.2f.',
            id1, id2, i1, i2, dist)

    return pairs_to_merge
