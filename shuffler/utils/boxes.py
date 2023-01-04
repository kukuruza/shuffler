import logging
import numpy as np
import cv2


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


def getIoU(roi1, roi2):
    ' Computes intersection over union for two rectangles. '
    intersection_y = max(0, (min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])))
    intersection_x = max(0, (min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])))
    intersection = intersection_x * intersection_y
    area1 = (roi1[3] - roi1[1]) * (roi1[2] - roi1[0])
    area2 = (roi2[3] - roi2[1]) * (roi2[2] - roi2[0])
    union = area1 + area2 - intersection
    IoU = intersection / float(union) if union > 0 else 0.
    return IoU


def expandRoi(roi, perc):
    '''
    Expands a ROI. Floats are rounded to the nearest integer.
    Args:
      roi:   list or tuple [y1, x1, y2, x2]
      perc:  tuple (perc_y, perc_x). Both must be > -0.5.
    '''
    roi = list(roi)
    perc_y, perc_x = perc
    if (perc_y, perc_x) == (0, 0):
        return roi
    if perc_y < -0.5 or perc_x < -0.5:
        raise ValueError('perc_y=%f and perc_x=%f must be > -0.5', perc_y,
                         perc_x)

    half_delta_y = float(roi[2] - roi[0]) * perc_y / 2
    half_delta_x = float(roi[3] - roi[1]) * perc_x / 2
    # expand each side
    roi[0] -= half_delta_y
    roi[1] -= half_delta_x
    roi[2] += half_delta_y
    roi[3] += half_delta_x
    return roi


def expandPolygon(xs, ys, perc):
    '''
    Expand polygon from its median center in all directions.
    Floating-point numbers are then rounded to the nearest integer.
    Args:
      xs:    A list of x values.
      ys:    A list of y values.
      perc:  A tuple of (perc_y, perc_x). Both values are float from -1 to inf.
    Returns:
      xs:    A list of x values.
      ys:    A list of y values.
    '''
    perc_y, perc_x = perc
    center_x = np.array(xs, dtype=float).mean()
    center_y = np.array(ys, dtype=float).mean()
    xs = [int(center_x + (x - center_x) * (1 + perc_x)) for x in xs]
    ys = [int(center_y + (y - center_y) * (1 + perc_y)) for y in ys]
    return xs, ys


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
    return roi


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
        roi = expandRoiUpToRatio(roi, target_ratio)

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
