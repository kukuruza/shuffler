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
    IoU = intersection / union if union > 0 else 0.
    return IoU


def expandRoiBorder(roi, imsize, perc, integer_result=True):
    '''
    Expands a ROI, and clips it within borders.
    Floats are rounded to the nearest integer.
    '''
    imheight, imwidth = imsize
    perc_y, perc_x = perc
    if (perc_y, perc_x) == (0, 0): return roi

    half_delta_y = float(roi[2] + 1 - roi[0]) * perc_y / 2
    half_delta_x = float(roi[3] + 1 - roi[1]) * perc_x / 2
    # the result must be within (imheight, imwidth)
    bbox_height = roi[2] + 1 - roi[0] + half_delta_y * 2
    bbox_width = roi[3] + 1 - roi[1] + half_delta_x * 2
    if bbox_height > imheight or bbox_width > imwidth:
        logging.warning(
            'expanded bbox of size (%d,%d) does not fit into image (%d,%d)' %
            (bbox_height, bbox_width, imheight, imwidth))
        # if so, decrease half_delta_y, half_delta_x
        coef = min(imheight / bbox_height, imwidth / bbox_width)
        logging.warning('decreased bbox to (%d,%d)' %
                        (bbox_height, bbox_width))
        bbox_height *= coef
        bbox_width *= coef
        logging.warning('decreased bbox to (%d,%d)' %
                        (bbox_height, bbox_width))
        half_delta_y = (bbox_height - (roi[2] + 1 - roi[0])) * 0.5
        half_delta_x = (bbox_width - (roi[3] + 1 - roi[1])) * 0.5
    # and a small epsilon to account for floating-point imprecisions
    EPS = 0.001
    # expand each side
    roi[0] -= (half_delta_y - EPS)
    roi[1] -= (half_delta_x - EPS)
    roi[2] += (half_delta_y - EPS)
    roi[3] += (half_delta_x - EPS)
    # move to clip into borders
    if roi[0] < 0:
        roi[2] += abs(roi[0])
        roi[0] = 0
    if roi[1] < 0:
        roi[3] += abs(roi[1])
        roi[1] = 0
    if roi[2] > imheight - 1:
        roi[0] -= abs((imheight - 1) - roi[2])
        roi[2] = imheight - 1
    if roi[3] > imwidth - 1:
        roi[1] -= abs((imwidth - 1) - roi[3])
        roi[3] = imwidth - 1
    # check that now averything is within borders (bbox is not too big)
    assert roi[0] >= 0 and roi[1] >= 0, str(roi)
    assert roi[2] <= imheight - 1 and roi[3] <= imwidth - 1, str(roi)
    # make integer
    if integer_result:
        roi = [int(round(x)) for x in roi]
    return roi


def expandRoi(roi, perc, integer_result=True):
    ''' Expands a ROI. Floats are rounded to the nearest integer. '''
    roi = list(roi)
    perc_y, perc_x = perc
    if (perc_y, perc_x) == (0, 0): return roi

    half_delta_y = float(roi[2] + 1 - roi[0]) * perc_y / 2
    half_delta_x = float(roi[3] + 1 - roi[1]) * perc_x / 2
    # and a small epsilon to account for floating-point imprecisions
    EPS = 0.001
    # expand each side
    roi[0] -= (half_delta_y - EPS)
    roi[1] -= (half_delta_x - EPS)
    roi[2] += (half_delta_y - EPS)
    roi[3] += (half_delta_x - EPS)
    # make integer
    if integer_result:
        roi = [int(round(x)) for x in roi]
    return roi


def expandPolygon(xs, ys, perc, integer_result=True):
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


def expandRoiToRatioBorder(roi, imsize, expand_perc, ratio):
    ''' Expands a ROI to keep 'ratio', and maybe more, up to 'expand_perc' '''
    imheight, imwidth = imsize
    bbox = roi2bbox(roi)
    # adjust width and height to ratio
    height = float(roi[2] + 1 - roi[0])
    width = float(roi[3] + 1 - roi[1])
    if height / width < ratio:
        perc = ratio * width / height - 1
        roi = expandRoiBorder(roi, (imheight, imwidth), (perc, 0),
                              integer_result=False)
    else:
        perc = height / width / ratio - 1
        roi = expandRoiBorder(roi, (imheight, imwidth), (0, perc),
                              integer_result=False)
    # additional expansion
    perc = (1 + expand_perc) / (1 + perc) - 1
    if perc > 0:
        roi = expandRoiBorder(roi, (imheight, imwidth), (perc, perc),
                              integer_result=False)
    roi = [int(round(x)) for x in roi]
    return roi


def expandRoiToRatio(roi, expand_perc, ratio):
    '''Expands a ROI to keep 'ratio', and maybe more, up to 'expand_perc'
  '''
    bbox = roi2bbox(roi)
    # adjust width and height to ratio
    height = float(roi[2] + 1 - roi[0])
    width = float(roi[3] + 1 - roi[1])
    if height / width < ratio:
        perc = ratio * width / height - 1
        roi = expandRoi(roi, (perc, 0), integer_result=False)
    else:
        perc = height / width / ratio - 1
        roi = expandRoi(roi, (0, perc), integer_result=False)
    # additional expansion
    #perc = (1 + expand_perc) / (1 + perc) - 1
    #if perc > 0:
    roi = expandRoi(roi, (expand_perc, expand_perc))
    roi = [int(round(x)) for x in roi]
    return roi


def cropPatch(image, roi, edge, target_height, target_width):
    ''' Crop a patch from the image.
    Args:
      edge:   {'distort', 'constant', 'background', 'original'}
              'distort'     Crops out the ROI and resizes it to target shape
                            without regards to aspect ratio.
              'constant'    Pads zeros on the sides of ROI to match the target
                            aspect ratio, then crops and resizes.
              'background'  Resizes the ROI to aspect ratio (with along X or Y),
                            then resizes to match the target dimensions.
              'original'    Crops out ROI and does NOT resize. Target dimensions
                            are ignored.
      target_height and target_width:
                  Target dimensions. Must be specified if edge != 'original'.
    Returns:
      patch:      The cropped patch. Color or grayscale depending on the image.
      transform:  A numpy float array of shape (3,3) representing an affine
                  transform from a point in the original image to that point in
                  the cropped image.
    TODO: Perspective crops may be necessary in the future. In that case,
          actual cropping may be better via cv2 perspectiveTransform function.
          Then a new function cropPerspective would be necessary.
    '''
    grayscale = len(image.shape) == 2

    # Make a copy of ROI.
    roi = list(roi)
    logging.debug('Cropping with ROI: %s', str(roi))

    # Maybe expand the ROI.
    if edge == 'background':
        target_ratio = target_height / target_width
        roi = expandRoiToRatio(roi, 0.0, target_ratio)

    height, width = roi[2] - roi[0], roi[3] - roi[1]

    # Start with identity transform.
    transform = np.eye(3, 3, dtype=float)
    # Transform is set to match the bottom-left corner now.
    transform[0, 2] = -roi[0]
    transform[1, 2] = -roi[1]

    if edge == 'background':
        if grayscale:
            pads = ((height, height), (width, width))
        else:
            pads = ((height, height), (width, width), (0, 0))
        image = np.pad(image, pad_width=pads, mode='constant')
        roi = [
            roi[0] + height, roi[1] + width, roi[2] + height, roi[3] + width
        ]
        logging.debug('Cropping as "background" adjusted ROI to: %s', str(roi))

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

    # Crop the image.
    patch = image[roi[0]:roi[2], roi[1]:roi[3]]

    # Apply the pads for compensating for ROI out of boundaries.
    patch = np.pad(patch, pad_width=pads, mode='constant')

    if edge == 'constant':
        if target_height is None or target_width is None:
            raise RuntimeError(
                'When edge is not "original", '
                'both target_height and target_width are required.')
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
        transform[0, 2] += pads[0][0]
        transform[1, 2] += pads[1][0]

    if edge != 'original':
        # The transform is scaled on X and Y to match the top-right corner.
        transform[0, 0] = target_height / float(patch.shape[0])
        transform[1, 1] = target_width / float(patch.shape[1])
        transform[0, 2] *= transform[0, 0]
        transform[1, 2] *= transform[1, 1]
        patch = cv2.resize(patch, dsize=(target_width, target_height))

    return patch, transform
