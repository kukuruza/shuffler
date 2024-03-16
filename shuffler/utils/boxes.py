import logging
import numpy as np
import cv2
import numbers
import collections
from shapely.geometry import Polygon as ShapelyPolygon


def validateBbox(bbox):
    '''
    Args:     A sequence (e.g. list or tuple) of {x1, y1, width, height}.
              Required: width >= 0, height >= 0.
    Returns:  None
    Raises a ValueError or TypeError if detected a problem with the input.
    '''
    if not isinstance(bbox, collections.abc.Collection):
        raise TypeError('Need an sequence, got %s' % type(bbox))
    if not len(bbox) == 4:
        raise ValueError('Need 4 numbers, not %d.' % len(bbox))
    for x in bbox:
        if not isinstance(x, numbers.Number):
            raise TypeError('Each element must be a number, got %s' % x)
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError('Bbox %s has a negative width or height.' % bbox)


def validateRoi(roi):
    '''
    Args:     A sequence (e.g. list or tuple) of {y1, x1, y2, x2}.
              Required: x1 <= x2, y1 <= y2.
    Returns:  None
    Raises a ValueError or TypeError if detected a problem with the input.
    '''
    if not isinstance(roi, collections.abc.Collection):
        raise TypeError('Need an sequence, got %s' % type(roi))
    if not len(roi) == 4:
        raise ValueError('Need 4 numbers, not %d.' % len(roi))
    for x in roi:
        if not isinstance(x, numbers.Number):
            raise TypeError('Each element must be a number, got %s' % x)
    if roi[2] < roi[0] or roi[3] < roi[1]:
        raise ValueError('Roi %s has a negative width or height.' % roi)


def validatePolygon(yxs):
    '''
    Args:     an iterable with at least 3 elements, each element is a collection 
              of two numbers.
    Returns:  None
    Raises a ValueError or TypeError if detected a problem with the input.
    '''
    logging.debug('Validating polygon %s', yxs)
    if not isinstance(yxs, collections.abc.Iterable):
        raise TypeError('Yxs must be iterables, not %s.' % type(yxs))
    count = 0
    for yx in yxs:
        count += 1
        if not isinstance(yx, collections.abc.Collection) or len(yx) != 2:
            raise TypeError(
                'Each element in yxs must be a iterable with two elements, got %s.'
                % yx)
        if (not isinstance(yx[0], numbers.Number)
                or not isinstance(yx[1], numbers.Number)):
            print(yx[0], yx[1], type(yx[0]), type(yx[1]))
            raise TypeError(
                'Elements in yxs must be a pair of numbers, not %s.' % yx)
    if count < 3:
        raise ValueError('Need at least 3 points, got yxs = %s.' % str(yxs))


def bbox2roi(bbox):
    '''
    Args:     [x1, y1, width, height]
    Returns:  [y1, x1, y2, x2]
    '''
    validateBbox(bbox)
    return [bbox[1], bbox[0], bbox[3] + bbox[1], bbox[2] + bbox[0]]


def roi2bbox(roi):
    '''
    Args:     [y1, x1, y2, x2]
    Returns:  [x1, y1, width, height]
    '''
    validateRoi(roi)
    return [roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0]]


def box2polygon(bbox):
    '''
    Args:     [x1, y1, w, h]
    Returns:  [(y1, x1), (y2, x2), (y3, x3), (y4, x4)]
    '''
    validateBbox(bbox)
    return [
        (bbox[1], bbox[0]),  #
        (bbox[1] + bbox[3], bbox[0]),
        (bbox[1] + bbox[3], bbox[0] + bbox[2]),
        (bbox[1], bbox[0] + bbox[2])
    ]


def getIoURoi(roi1, roi2):
    ' Computes intersection over union for two rectangles. '
    intersection_y = max(0, (min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])))
    intersection_x = max(0, (min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])))
    intersection = intersection_x * intersection_y
    area1 = (roi1[3] - roi1[1]) * (roi1[2] - roi1[0])
    area2 = (roi2[3] - roi2[1]) * (roi2[2] - roi2[0])
    union = area1 + area2 - intersection
    IoU = intersection / float(union) if union > 0 else 0.
    return IoU


def getIoUPolygons(polygons):
    '''
    Computes intersection-over-union for several polygons.
    Args:
      polygons1:    a list of polygons, where each polygon is
                    [(y1, x1), (y2, x2), (y3, x3), ...]
    Returns:
      A float in range [0, 1].
    '''
    if len(polygons) == 0:
        raise ValueError('getIoUPolygons got an empty polygon: %s.', polygons2)

    polygons = [
        ShapelyPolygon(yxs) for yxs in polygons if validatePolygon(yxs) is None
    ]
    intersection = polygons[0]
    union = polygons[0]

    for p in polygons[1:]:
        intersection = intersection.intersection(p)
        union = union.union(p)

    logging.debug('intersection: %f, union: %f', intersection.area, union.area)
    return intersection.area / float(union.area) if union.area > 0 else 0.


def getIntersectingPolygons(polygons_by_objectids: dict,
                            IoU_threshold: float) -> list:
    '''
    Merge polygons into clusters based in IoU.

    Args:
      polygons_by_objectid: A dict {objectid: [polygon, polygon, ...]},
                            where each polygon is [(y,x), (y,x), ...]
      IoU_threshold:        A float in range (0, 1].
    Returns:
      A list of sets.       Each set contains objectids to merge.
    '''
    assert IoU_threshold > 0. and IoU_threshold <= 1., IoU_threshold

    N = len(polygons_by_objectids)
    if len(polygons_by_objectids) == 0:
        return []

    # Init clusters, intersections, and unions.
    clusters = []
    intersections = []
    unions = []
    for objectid, polygons in polygons_by_objectids.items():
        clusters.append([objectid])
        if len(polygons) == 0:
            raise ValueError('Objectid %d came with no polygons' % objectid)
        validatePolygon(polygons[0])
        intersection = ShapelyPolygon(polygons[0])
        union = intersection
        for yxs in polygons[1:]:
            validatePolygon(yxs)
            polygon = ShapelyPolygon(yxs)
            intersection = intersection.intersection(polygon)
            union = union.union(polygon)
        intersections.append(intersection)
        unions.append(union)

    # Fill in pairwise_IoU.
    pairwise_IoU = np.full(shape=(N, N), fill_value=-np.inf, dtype=float)
    for i in range(N):
        for j in range(N):
            if i < j:
                intersection_area = intersections[i].intersection(
                    intersections[j]).area
                union_area = unions[i].union(unions[j]).area
                pairwise_IoU[i, j] = intersection_area / union_area
    logging.debug('Initial pairwise_IoU:\n%s', pairwise_IoU)

    # Agglomerative merging.
    while N > 1:

        # Find the highest IoU.
        i0, j0 = np.unravel_index(np.argmax(pairwise_IoU), pairwise_IoU.shape)
        assert i0 < j0, 'Can not happen, pairwise matrix is upper-triangular.'
        IoU = pairwise_IoU[i0, j0]
        logging.debug('Merging clusters %d and %d. IoU: %s', i0, j0, str(IoU))

        # Stop if no more good pairs.
        if IoU < IoU_threshold:
            break

        # Update polygons of intersections and unions.
        intersections[i0] = intersections[i0].intersection(intersections[j0])
        unions[i0] = unions[i0].union(unions[j0])
        del intersections[j0]
        del unions[j0]
        N = N - 1

        # Merge objectids in the two clusters.
        clusters[i0] += clusters[j0]
        del clusters[j0]
        logging.debug('New clusters:\n%s', clusters)

        # Update pairwise_IoU.
        pairwise_IoU = np.delete(pairwise_IoU, j0, axis=0)
        pairwise_IoU = np.delete(pairwise_IoU, j0, axis=1)
        for i in range(N):
            if i == i0:
                continue
            intersection_area = intersections[i].intersection(
                intersections[i0]).area
            union_area = unions[i].union(unions[i0]).area
            IoU = intersection_area / union_area
            if i < i0:
                pairwise_IoU[i, i0] = IoU
            else:
                pairwise_IoU[i0, i] = IoU
        logging.debug('New pairwise_IoU:\n%s', pairwise_IoU)

    logging.debug('Found %d clusters: %s', len(clusters), clusters)
    return [tuple(sorted(cluster)) for cluster in clusters]


def clipPolygonToRoi(yxs, roi):
    '''
    If the polygon intersects with ROI, each polygon point is clipped to ROI.
    If the polygon is inside, is guaranteed to return the original polygon.
    If the polygon is outside, an empty list is returned.
    Args:
      yxs:  [(y1, x1), (y2, x2), (y3, x3), ...]
      roi:  [y1, x1, y2, x2]
    Returns:
      [(y1, x1), (y2, x2), (y3, x3), ...]
    '''
    validatePolygon(yxs)
    validateRoi(roi)

    p1 = ShapelyPolygon(yxs)
    p2 = ShapelyPolygon([(roi[0], roi[1]), (roi[0], roi[3]), (roi[2], roi[3]),
                         (roi[2], roi[1])])

    p = p1.intersection(p2)
    logging.debug(p)

    intersection_area = p1.intersection(p2).area
    if intersection_area == 0:
        logging.debug('Polygon %s and roi %s do not intersect.', yxs, roi)
        return []
    elif intersection_area == p1.area:
        logging.debug('Polygon %s is inside roi %s.', yxs, roi)
        return yxs
    else:
        return [(max(roi[0], min(y, roi[2])), max(roi[1], min(x, roi[3])))
                for y, x in yxs]


def intersectPolygonAndRoi(yxs, roi):
    '''
    Computes and returns the intersection of a polygon with a rectange.
    If the polygon is inside, is guaranteed to return the original polygon.
    Args:
      yxs:  [(y1, x1), (y2, x2), (y3, x3), ...]
      roi:  [y1, x1, y2, x2]
    Returns:
      [(y1, x1), (y2, x2), (y3, x3), ...]
    '''
    validatePolygon(yxs)
    validateRoi(roi)

    p1 = ShapelyPolygon(yxs)
    p2 = ShapelyPolygon([(roi[0], roi[1]), (roi[0], roi[3]), (roi[2], roi[3]),
                         (roi[2], roi[1])])

    p = p1.intersection(p2)
    logging.debug(p)

    intersection_area = p1.intersection(p2).area
    if intersection_area == 0:
        logging.debug('Polygon %s and roi %s do not intersect.', yxs, roi)
        return []
    elif intersection_area == p1.area:
        logging.debug('Polygon %s is inside roi %s.', yxs, roi)
        return yxs

    return list(p.exterior.coords[:-1])


def expandRoi(roi, perc):
    '''
    Expands a ROI. Floats are rounded to the nearest integer.
    Args:
      roi:   list or tuple [y1, x1, y2, x2]
      perc:  tuple (perc_y, perc_x). Both must be > -0.5.
             perc=1. means that height and width will be doubled.
    '''
    roi = list(roi)
    perc_y, perc_x = perc
    if (perc_y, perc_x) == (0, 0):
        return roi
    if perc_y < -0.5 or perc_x < -0.5:
        raise ValueError('perc_y=%f and perc_x=%f must be > -0.5' %
                         (perc_y, perc_x))

    half_delta_y = float(roi[2] - roi[0]) * perc_y / 2
    half_delta_x = float(roi[3] - roi[1]) * perc_x / 2
    # expand each side
    roi[0] -= half_delta_y
    roi[1] -= half_delta_x
    roi[2] += half_delta_y
    roi[3] += half_delta_x
    return roi


def expandPolygon(ys, xs, perc):
    '''
    Expand polygon from its avg(ymin, ymax), avg(xmin, xmax) in all directions.
    Args:
      ys:    list of float y values.
      xs:    list of float x values.
      perc:  tuple (perc_y, perc_x). Both must be > -0.5.
             perc=1. means that height and width will be doubled.
    Returns:
      ys:    list of float y values.
      xs:    list of float x values.
    '''
    if isinstance(ys, np.ndarray) and isinstance(xs, np.ndarray):
        validatePolygon(np.stack((ys, xs)).transpose())
    else:
        validatePolygon(zip(ys, xs))

    perc_y, perc_x = perc
    if (perc_y, perc_x) == (0, 0):
        return ys, xs
    if perc_y < -0.5 or perc_x < -0.5:
        raise ValueError('perc_y=%f and perc_x=%f must be > -0.5' %
                         (perc_y, perc_x))
    ys = np.array(ys, dtype=float)
    xs = np.array(xs, dtype=float)
    center_y = (ys.min() + ys.max()) / 2
    center_x = (xs.min() + xs.max()) / 2
    logging.info('Center: %s', (center_x, center_y))
    ys = [center_y + (y - center_y) * (1 + perc_y) for y in ys]
    xs = [center_x + (x - center_x) * (1 + perc_x) for x in xs]
    return ys, xs


def expandRoiUpToRatio(roi, ratio):
    '''Expands a ROI to match 'ratio'. '''
    # adjust width and height to ratio
    height = float(roi[2] - roi[0])
    width = float(roi[3] - roi[1])
    if height / width < ratio:
        perc = ratio * width / height - 1
        roi = expandRoi(roi, (perc, 0))
    else:
        perc = height / width / ratio - 1
        roi = expandRoi(roi, (0, perc))
    return roi, perc


def expandPolygonUpToRatio(ys, xs, ratio):
    '''Expands a polygon to match 'ratio'. '''
    ys = np.array(ys, dtype=float)
    xs = np.array(xs, dtype=float)
    height = ys.min() - ys.max()
    width = xs.min() - xs.max()

    # adjust width and height to ratio
    if height / width < ratio:
        perc = ratio * width / height - 1
        ys, xs = expandPolygon(ys, xs, (perc, 0))
    else:
        perc = height / width / ratio - 1
        ys, xs = expandPolygon(ys, xs, (0, perc))
    return ys, xs, perc


def cropPatch(image, roi, edge, target_height, target_width):
    ''' Crop a patch from the image.
    Args:
      edge:   {'distort', 'constant', 'background', 'original'}
              'distort'     Crops out the ROI and resizes it to target shape
                            without regards to aspect ratio.
              'constant'    Pads zeros (black) on the sides of ROI to match
                            the target aspect ratio, then crops and resizes.
              'background'  Resizes the ROI to aspect ratio (with along X or Y),
                            then resizes to match the target dimensions. Stuff
                            outside of the image is black.
              'original'    Crops out ROI and does NOT resize. Target dimensions
                            are ignored.
      target_height and target_width:
                  Target dimensions. Must be specified if edge != 'original'.
    Returns:
      patch:      The cropped patch. Color or grayscale depending on the image.
                  Non-integer ROI is reduced to the nearest int on each side.
      transform:  A numpy float array of shape (3,3) representing an affine
                  transform from a point in the original image to that point in
                  the cropped image. Applying transform to an integer ROI gives
                  [0, 0, H, W] where HxW is the size of the crop. Applying it
                  to NON-integer ROI produces [-e1, -e2, H + e3, W + e4], where
                  "eps" is a small number.
    TODO: Perspective crops may be necessary in the future. In that case,
          actual cropping may be better via cv2 perspectiveTransform function.
          Then a new function cropPerspective would be necessary.
    '''
    logging.debug('Cropping with ROI: %s', str(roi))

    if (edge != 'original' and
        (target_height is None or not isinstance(target_height, int)
         or target_width is None or not isinstance(target_width, int))):
        raise RuntimeError(
            'When edge is not "original", target_height and target_width are '
            'required and must be int, not %s, %s' %
            (target_height, target_width))

    grayscale = len(image.shape) == 2

    # Make a deep copy.
    roi = list(roi)

    # Maybe expand the ROI.
    if edge == 'background':
        target_ratio = target_height / target_width
        roi, _ = expandRoiUpToRatio(roi, target_ratio)

    # Reduce the bbox to the nearest integer pixel in every direction.
    roi[0] = int(np.ceil(roi[0]))
    roi[1] = int(np.ceil(roi[1]))
    roi[2] = int(np.floor(roi[2]))
    roi[3] = int(np.floor(roi[3]))

    height, width = roi[2] - roi[0], roi[3] - roi[1]
    if height <= 1 or width <= 1:
        raise ValueError('Cant crop from HxW = %dx%d.' % (height, width))

    # Transform from the original bbox to the cooridnates of the crop.
    # If bbox was non-integer, the transformed roi is [-e1, -e2, H+e3, W+e4],
    #   where "e" is small, H & W are dimensions of the crop.
    transform = np.eye(3, 3, dtype=float)
    transform[0, 2] = -roi[0]
    transform[1, 2] = -roi[1]
    logging.debug('Transform of taking ROI to origin:\n%s', transform)

    # Pad parts of the rois out of the image boundaries with zero.
    padsy = max(0, -roi[0]), max(0, roi[2] - image.shape[0])
    padsx = max(0, -roi[1]), max(0, roi[3] - image.shape[1])
    if grayscale:
        pads = (padsy, padsx)
    else:
        pads = (padsy, padsx, (0, 0))
    logging.debug('Pads for compensating ROI out of boundaries: %s', str(pads))
    roi[0] += padsy[0]
    roi[1] += padsx[0]
    roi[2] -= padsy[1]
    roi[3] -= padsx[1]
    logging.debug('Roi with pads compensated for out of boundaries: %s', roi)

    # Crop the image.
    patch = image[roi[0]:roi[2], roi[1]:roi[3]]

    # Apply the pads for compensating for ROI out of boundaries.
    patch = np.pad(patch, pad_width=pads, mode='constant')

    if edge == 'constant':

        target_ratio = target_height / target_width
        if height > int(target_ratio * width):
            pad1 = (int(height / target_ratio) - width) // 2
            pad2 = int(height / target_ratio) - width - pad1
            if grayscale:
                pads = ((0, 0), (pad1, pad2))
            else:
                pads = ((0, 0), (pad1, pad2), (0, 0))
        else:
            pad1 = (int(target_ratio * width) - height) // 2
            pad2 = int(target_ratio * width) - height - pad1
            if grayscale:
                pads = ((pad1, pad2), (0, 0))
            else:
                pads = ((pad1, pad2), (0, 0), (0, 0))

        logging.debug(
            'Padding as "constant" (target_ratio: %.3f) with pad %s'
            ' to shape %s.', target_ratio, str(pads), str(patch.shape))
        patch = np.pad(patch, pad_width=pads, mode='constant')
        # Transform is offset to match the bottom-left corner now.
        transform_pad = np.eye(3, 3, dtype=float)
        transform_pad[0, 2] = pads[0][0]
        transform_pad[1, 2] = pads[1][0]
        transform = np.dot(transform_pad, transform)
        logging.debug('Transform for padding:\n%s', transform_pad)
        logging.debug('Combined transform after padding:\n%s', transform)

    if edge != 'original':
        # The transform is scaled on X and Y to match the top-right corner.
        transform_scale = np.eye(3, 3, dtype=float)
        transform_scale[0, 0] = target_height / float(patch.shape[0])
        transform_scale[1, 1] = target_width / float(patch.shape[1])
        transform = np.dot(transform_scale, transform)
        logging.debug('Transform for scaling patch:\n%s', transform_scale)
        logging.debug('Combined transform after scaling patch:\n%s', transform)

        patch = cv2.resize(patch, dsize=(target_width, target_height))

    return patch, transform


def applyTransformToRoi(transform, roi):
    '''
    Apply tranform that Shuffler uses to the provided ROI.
    Args:
      roi:        (y1, x1, y2, x2)
      transform   np array of shape 2x3. roi_new = transform * roi.
    '''
    (y1, x1, y2, x2) = roi
    roi = np.array([[y1, y2], [x1, x2], [1., 1.]])
    roi = np.dot(transform, roi)
    roi = (roi[0, 0], roi[1, 0], roi[0, 1], roi[1, 1])
    logging.debug(roi)
    return roi


def clipRoiToShape(roi, shape):
    return (max(roi[0], 0), max(roi[1],
                                0), min(roi[2],
                                        shape[0]), min(roi[3], shape[1]))


def _getTransformBetweenRois(roi_from, roi_to):
    if roi_from[2] <= roi_from[0] or roi_from[3] <= roi_from[1]:
        raise ValueError('Roi_from has weight or height <= 0: %s' % roi_from)
    if roi_to[2] <= roi_to[0] or roi_to[3] <= roi_to[1]:
        raise ValueError('Roi_to has weight or height <= 0: %s' % roi_to)

    transform = np.eye(3, 3, dtype=float)
    transform[0, 0] = (roi_to[2] - roi_to[0]) / (roi_from[2] - roi_from[0])
    transform[1, 1] = (roi_to[3] - roi_to[1]) / (roi_from[3] - roi_from[1])
    transform[0, 2] = roi_to[0] - transform[0, 0] * roi_from[0]
    transform[1, 2] = roi_to[1] - transform[1, 1] * roi_from[1]
    return transform
