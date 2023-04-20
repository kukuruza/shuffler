import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

from shuffler.utils import general as general_utils

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 28
FONT_SIZE = 1.2
THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_BACKCOLOR = (0, 0, 0)


def _drawFilledRoi(img, roi, color, fill_opacity):
    ''' 
    Draw a filled rectangle in the image. 
    Correctly processes ROI out of image boundary.
    Args:
      img:       Color or grayscale
      polygon:   [y1, x1, y2, x2]
      color:     Tuple of length 3 OR int based on whether image is grayscale.
    '''
    assert (
        len(img.shape) == 3 and len(color) == 3
        or len(img.shape) == 2 and isinstance(color, int)
    ), 'Expect color image and RGB color or grayscale image and INT color.'
    is_color = len(img.shape) == 3

    roi = [int(a) for a in roi]
    roi[0] = max(roi[0], 0)
    roi[1] = max(roi[1], 0)
    roi[2] = min(roi[2], img.shape[0])
    roi[3] = min(roi[3], img.shape[1])
    shape = roi[2] - roi[0], roi[3] - roi[1], (3 if is_color else 1)

    sub_img = img[roi[0]:roi[2], roi[1]:roi[3]]
    sub_filled = np.zeros(shape, dtype=np.uint8)
    sub_filled[:] = color

    assert sub_img.shape == np.squeeze(sub_filled).shape, (
        "Roi %s created a misshaped rect: sub_img.shape=%s vs sub_filled.shape=%s"
        % (roi, sub_img.shape, sub_filled.shape))
    sub_overlayed = cv2.addWeighted(sub_img, (1. - fill_opacity), sub_filled,
                                    fill_opacity, 1.0)

    # Putting the image back to its position
    logging.debug('_drawFilledRoi roi: %s', roi)
    img[roi[0]:roi[2], roi[1]:roi[3]] = sub_overlayed


def _drawFilledPolygon(img, polygon, color, fill_opacity):
    '''
    Draw a filled rectangle in the image.
    Args:
      img:      Color or grayscale
      polygon:  [[y1, x1], [y2, x2], ...]
      color:    Tuple of length 3 OR int based on whether image is grayscale.
    '''
    assert (
        len(img.shape) == 3 and len(color) == 3
        or len(img.shape) == 2 and isinstance(color, int)
    ), 'Expect color image and RGB color or grayscale image and INT color.'
    is_color = len(img.shape) == 3

    # 0, 1, or 2 points in a polygon should not paint an area.
    if len(polygon) < 3:
        return

    polygon = np.array(polygon, np.int32)
    ymin = polygon[:, 0].min()
    xmin = polygon[:, 1].min()
    ymax = polygon[:, 0].max()
    xmax = polygon[:, 1].max()

    # Process polygons that are out of image boundary.
    if ymin < 0 or xmin < 0 or ymax >= img.shape[0] or xmax >= img.shape[1]:
        ymin_orig = ymin
        ymax_orig = ymax
        xmin_orig = xmin
        xmax_orig = xmax
        ymin = max(0, ymin)
        ymax = min(ymax, img.shape[0])
        xmin = max(0, xmin)
        xmax = min(xmax, img.shape[1])
        sub_img = img[ymin:ymax, xmin:xmax]

        # Pad the image so that the polygon is all within the image.
        pady = max(0, -ymin_orig), max(0, ymax_orig - img.shape[0])
        padx = max(0, -xmin_orig), max(0, xmax_orig - img.shape[1])
        pad = (pady, padx, (0, 0)) if is_color else (pady, padx)
        img_padded = np.pad(img, pad)
        # Crop from the padded image.
        polygon[:, 0] -= ymin_orig
        polygon[:, 1] -= xmin_orig
        sub_filled = img_padded[ymin_orig + pady[0]:ymax_orig + pady[0],
                                xmin_orig + padx[0]:xmax_orig + padx[0]]
        cv2.fillPoly(sub_filled, [polygon[:, ::-1]], color)
        sub_filled = sub_filled[pady[0]:ymax - ymin + pady[0],
                                padx[0]:xmax - xmin + padx[0]]
    else:
        sub_img = img[ymin:ymax, xmin:xmax]
        sub_filled = sub_img.copy()
        polygon[:, 0] -= ymin
        polygon[:, 1] -= xmin
        cv2.fillPoly(sub_filled, [polygon[:, ::-1]], color)

    assert sub_img.shape == sub_filled.shape, (
        "Polygon %s created a misshaped rect: sub_img.shape=%s vs sub_filled.shape=%s"
        % (polygon, sub_img.shape, sub_filled.shape))
    sub_overlayed = cv2.addWeighted(sub_img, (1 - fill_opacity), sub_filled,
                                    fill_opacity, 1.0)

    # Putting the image back to its position
    img[ymin:ymax, xmin:xmax] = sub_overlayed


def drawScoredRoi(img,
                  roi,
                  label=None,
                  score=None,
                  fill_opacity=0.,
                  colormap='inferno_r',
                  **kwargs):
    '''
    Draw a bounding box on top of image.
    Args:
      img:      Numpy color image.
      roi:      List/tuple [y1, x1, y2, x2].
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
      fill_opacity:     If > 0, fill in the bounding box with specified opacity.
      colormap: Name of the Matplotlib colormap.
      kwargs:   Ignored.
    Return:
      Nothing, 'img' is changed in place.
    '''
    assert len(img.shape) == 3 and img.shape[2] == 3, (
        'Image must be color, but is grayscale: %s' % str(img.shape))
    assert score is None or score >= 0 and score <= 1
    if label is None:
        label = ''
    label = general_utils.maybeDecode(label)
    if score is None:
        score = 1
    color = tuple([
        int(x * 255) for x in plt.cm.get_cmap(colormap)(float(score))
    ][0:3][::-1])
    roi = [int(a) for a in roi]

    text_coords = (roi[1], roi[0] - SCALE)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_COLOR,
                THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_BACKCOLOR,
                THICKNESS - 1)

    if fill_opacity > 0:
        _drawFilledRoi(img, roi, color, fill_opacity)
    cv2.rectangle(img, (roi[1], roi[0]), (roi[3], roi[2]), color, THICKNESS)


def drawScoredPolygon(img,
                      polygon,
                      label=None,
                      score=None,
                      fill_opacity=0.,
                      colormap='inferno_r',
                      **kwargs):
    '''
    Args:
      img:      Numpy color image.
      polygon:  [[y1, x1], [y2, x2], ...]
      label:    String to print near the bounding box or None.
      score:    A float in range [0, 1] or None.
      fill_opacity:     If > 0, fill in the bounding box with specified opacity.
      colormap: Name of the Matplotlib colormap.
      kwargs:   Ignored.
    Returns:
      Nothing, 'img' is changed in place.
    '''
    assert len(img.shape) == 3 and img.shape[2] == 3, (
        'Image must be color, but is grayscale: %s' % str(img.shape))
    if label is None:
        label = ''
    label = general_utils.maybeDecode(label)
    if score is None:
        score = 1
    color = tuple([
        int(x * 255) for x in plt.cm.get_cmap(colormap)(float(score))
    ][0:3][::-1])
    polygon = np.array(polygon, np.int32)

    # Draw a label.
    ymin = polygon[:, 0].min()
    xmin = polygon[:, 1].min()
    text_coords = (xmin, ymin - SCALE)
    logging.debug('polygon:\n%s', polygon)
    logging.debug('text_coords: %s', text_coords)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_COLOR,
                THICKNESS)
    cv2.putText(img, label, text_coords, FONT, FONT_SIZE, TEXT_BACKCOLOR,
                THICKNESS - 1)

    if fill_opacity > 0:
        _drawFilledPolygon(img, polygon, color, fill_opacity)
    cv2.polylines(img, [polygon[:, ::-1]], True, color, THICKNESS)


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
        mask = general_utils.applyMaskMapping(mask, labelmap).astype(np.uint8)

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
        mask = general_utils.applyMaskMapping(mask, labelmap)

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
