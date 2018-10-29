import os, os.path as op
import shutil
import logging
import cv2
import numpy as np
from pprint import pformat
import simplejson as json
import matplotlib.pyplot as plt  # for colormaps


FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 28
FONT_SIZE = 0.7
THICKNESS = 3


def copyWithBackup (in_path, out_path):
  ''' Copy in_path into out_path, which is backed up if already exists. '''

  if not op.exists (in_path):
    raise FileNotFoundError ('File does not exist: "%s"' % in_path)
  if op.exists (out_path):
    logging.warning ('Will back up existing out_path "%s"' % out_path)
    ext = op.splitext(out_path)[1]
    backup_path = op.splitext(out_path)[0]  + '.backup%s' % ext
    if in_path != out_path:
      if op.exists (backup_path):
        os.remove (backup_path)
      os.rename (out_path, backup_path)
    else:
      shutil.copyfile(in_path, backup_path)
  if in_path != out_path:
    # copy input file into the output one
    shutil.copyfile(in_path, out_path)


def bbox2roi (bbox):
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
      raise TypeError('Each element must be a number, but got %s' % type(x))
  if bbox[2] < 0 or bbox[3] < 0:
    raise ValueError('Bbox %s has negative width or height.' % str(bbox))
  return [bbox[1], bbox[0], bbox[3]+bbox[1], bbox[2]+bbox[0]]


def roi2bbox (roi):
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
      raise TypeError('Each element must be a number, but got %s' % type(x))
  if roi[2] < roi[0] or roi[3] < roi[1]:
    raise ValueError('Roi %s has negative width or height.' % str(roi))
  return [roi[1], roi[0], roi[3]-roi[1], roi[2]-roi[0]]


def overlapRatio (roi1, roi2):
    assert (len(roi1) == 4 and len(roi2) == 4)
    if roi1 == roi2: return 1  # same object
    dy = min(roi1[2], roi2[2]) - max(roi1[0], roi2[0])
    dx = min(roi1[3], roi2[3]) - max(roi1[1], roi2[1])
    if dy <= 0 or dx <= 0: return 0
    area1 = (roi1[2] - roi1[0]) * (roi1[3] - roi1[1])
    area2 = (roi2[2] - roi2[0]) * (roi2[3] - roi2[1])
    inters = dx * dy
    union  = area1 + area2 - inters
    logging.debug('inters: ' + str(inters) + ', union: ' +  str(union))
    assert (union >= inters and inters > 0)
    return float(inters) / union


def drawScoredPolygon (img, polygon, label=None, score=None):
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
  if score is None:
    score = 1
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
  for i1 in range(len(polygon)):
    i2 = (i1 + 1) % len(polygon)
    cv2.line(img, tuple(polygon[i1]), tuple(polygon[i2]), color, THICKNESS)
  # Draw a label.
  xmin = polygon[0][0]
  ymin = polygon[0][1]
  for i in range(len(polygon)-1):
    xmin = min(xmin, polygon[i][0])
    ymin = min(ymin, polygon[i][1])
  cv2.putText (img, label, (xmin, ymin - SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, label, (xmin, ymin - SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)


def drawScoredRoi (img, roi, label=None, score=None):
  assert score is None or score >= 0 and score <= 1
  if label is None: label = ''
  if score is None:
    score = 1
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
  cv2.rectangle (img, (roi[1], roi[0]), (roi[3], roi[2]), color, THICKNESS)
  cv2.putText (img, label, (roi[1], roi[0] - SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, label, (roi[1], roi[0] - SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)


def drawImageId(img, imagefile):
  '''
  Draw the image name in the corner of the image.
  Returns:
    Nothing. Input "img" is changed in place.
  '''
  img = img.copy()  # If order was reversed.
  imheight, imwidth = img.shape[0:2]
  fontscale = float(imheight) / 700
  thickness = min(imheight // 700, 1)
  offsety = imheight // 30
  imageid = op.basename(imagefile)
  cv2.putText (img, imageid, (10, SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, imageid, (10, SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)


def applyLabelMappingToMask(mask, labelmap):
  '''
  Args:
    mask:     numpy 2-dim array, the same HxW as img.
    labelmap: dict from mask values to output values Key->Value, where:
              Key is scalar int in [0, 255].
              If Value is a scalar, the mask stays grayscale,
              if Value is a tuple (r,g,b), the mask is transformed to color.
              Examples {1: 255, 2: 128} or {1: (255,255,255), 2: (128,255,0)}.
              If not specified, the mask is used as is.
  Returns:
    mask      Mapped mask.
  '''
  if len(mask.shape) == 3:
    raise NotImplementedError('Color masks are not supported now.')

  logging.debug('labelmap: %s' % labelmap)
  if len(labelmap) is 0:
    raise ValueError('"labelmap" is empty.')

  dmask = None
  for key, value in labelmap.items():
    # Lazy initialization of dmask.
    if dmask is None:
      if isinstance(value, (list, tuple)) and len(value) == 3:
        # Create a dmask of shape (3, H, W)
        dmask = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
      elif isinstance(value, int):
        # Create a dmask of shape (H, W)
        dmask = np.zeros(mask.shape[0:2], dtype=np.uint8)
      else:
        raise TypeError('Values of "labelmap" are neither a collection of length 3, not a number.')
    # For grayscale target.
    if len(dmask.shape) == 2 and isinstance(value, int):
      dmask[mask == key] = value
    # For color target.
    elif len(dmask.shape) == 3 and isinstance(value, (list, tuple)) and len(value) == 3:
      dmask[0][mask == key] = value[0]
      dmask[1][mask == key] = value[1]
      dmask[2][mask == key] = value[2]
    else:
      raise ValueError('"labelmap" value %s mismatches dmask\'s shape %s' % (value, dmask.shape))
    logging.debug('key: %d, value: %s, numpixels: %d.' %
      (key, value, np.count_nonzero(mask == key)))
  if len(dmask.shape) == 3:
    # From (3, H, W) to (H, W, 3)
    dmask = np.transpose(dmask, (1, 2, 0))
  mask = dmask



def drawMaskOnImage(img, mask, alpha=0.5, labelmap=None):
  '''
  Draw a mask on the image, with colors.
  Args:
    img:      numpy array
    mask:     numpy 2-dim array, the same HxW as img.
    labelmap: same as arguments of applyLabelMappingToMask
    alpha:    float, alpha=0 means mask is not visible, alpha=1 means img is not visible.
  Returns:
    Output image.
  '''
  if not len(img.shape) == 3:
    raise NotImplementedError('Only color images are supported now.')

  if labelmap is not None:
    mask = applyLabelMappingToMask(mask, labelmap)

  if len(mask.shape) == 2:
    mask = cv2.cvtColor (mask, cv2.COLOR_GRAY2RGB)
  if img.shape[:2] != mask.shape[:2]:
    logging.warning('Image shape %s mismatches mask shape %s.' % (img.shape, mask.shape))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
  assert img.shape == mask.shape, (img.shape, mask.shape)

  # Overlay mask.
  logging.debug('alpha %.3f' % alpha)
  img = cv2.addWeighted(src1=img, alpha=1-alpha, src2=mask, beta=alpha, gamma=0)
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

  if len(mask.shape) == 3 and mask.shape[2] == 4 and len(img.shape) == 3 and img.shape[2] == 3:
    logging.debug('Mask has alpha channel, but image does not. Add it to image.')
    img = np.concatenate([img, np.ones((img.shape[0],img.shape[1],1), dtype=np.uint8) * 255], axis=2)
  if len(mask.shape) == 2 and len(img.shape) == 3 and img.shape[2] == 3:
    mask = cv2.cvtColor (mask, cv2.COLOR_GRAY2RGB)
  if mask.dtype == np.uint16:  # 16 bit is transferred to 8 bit.
    mask = (mask // 256).astype(np.uint8)
  if img.shape[:2] != mask.shape[:2]:
    logging.warning('Image shape %s mismatches mask shape %s.' % (img.shape, mask.shape))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
  assert img.shape == mask.shape, (img.shape, mask.shape)

  # Draw mask aside.
  img = np.hstack([img, mask])
  return img


def cropPatch(image, roi, target_height, target_width, edge):
  ''' Crop a patch from the image.
  Args:
    edge:          {'distort', 'constant', 'expand'}
    target_ratio:  height / width
  '''
  grayscale = len(image.shape) == 2

  if edge == 'background':
    target_ratio = target_height / target_width
    roi = expandRoiToRatio (roi, 0.0, target_ratio)

  pad = image.shape[1]
  if grayscale:
    pad_width=((pad,pad),(pad,pad))
  else:
    pad_width=((pad,pad),(pad,pad),(0,0))
  image = np.pad(image, pad_width=pad_width, mode='constant')
  roi = [x + pad for x in roi]
  height, width = roi[2] - roi[0], roi[3] - roi[1]

  if grayscale:
    patch = image[roi[0]:roi[2], roi[1]:roi[3]]
  else:
    patch = image[roi[0]:roi[2], roi[1]:roi[3], :]

  if edge == 'constant':
    target_ratio = target_height / target_width
    if height > int(target_ratio * width):
      pad1 = (int(height / target_ratio) - width) // 2
      pad2 = int(height / target_ratio) - width - pad1
      if grayscale:
        pad_width=((0,0),(pad1,pad2))
      else:
        pad_width=((0,0),(pad1,pad2),(0,0))
      patch = np.pad(patch, pad_width=pad_width, mode='constant')
    else:
      pad1 = (int(target_ratio * width) - height) // 2
      pad2 = int(target_ratio * width) - height - pad1
      if grayscale:
        pad_width=((pad1,pad2),(0,0))
      else:
        pad_width=((pad1,pad2),(0,0),(0,0))
      patch = np.pad(patch, pad_width=pad_width, mode='constant')

  elif edge == 'distort':
    pass

  patch = cv2.resize(patch, dsize=(target_width, target_height))
  return patch
