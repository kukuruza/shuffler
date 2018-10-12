import os, os.path as op
import shutil
import logging
import cv2
import numpy as np
from pprint import pformat
import simplejson as json
from collections import OrderedDict
import matplotlib.pyplot as plt  # for colormaps


FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 28
FONT_SIZE = 0.7
THICKNESS = 3


def copyWithBackup (in_path, out_path):
  ''' Copy in_path into out_path, which is backed up if already exists. '''

  if not op.exists (in_path):
    raise Exception ('File does not exist: "%s"' % in_path)
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
    assert ((isinstance(bbox, list) or isinstance(bbox, tuple)) and len(bbox) == 4)
    return [bbox[1], bbox[0], bbox[3]+bbox[1]-1, bbox[2]+bbox[0]-1]

def roi2bbox (roi):
    assert ((isinstance(roi, list) or isinstance(roi, tuple)) and len(roi) == 4)
    return [roi[1], roi[0], roi[3]-roi[1]+1, roi[2]-roi[0]+1]


def drawScoredPolygon (img, polygon, label=None, score=None):
  '''
  Args:
    polygon:  a list of coordinates (x,y)
  '''
  assert len(polygon) > 2, polygon
  
  if label is None: label = ''
  if score is None:
    score = 0
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
  for i1 in range(len(polygon)):
    i2 = (i1 + 1) % len(polygon)
    cv2.line(img, polygon[i1], polygon[i2], color, THICKNESS)
  # Draw a label.
  xmin = polygon[0][0]
  ymin = polygon[0][1]
  for i in range(len(polygon)-1):
    xmin = min(xmin, polygon[i][0])
    ymin = min(ymin, polygon[i][1])
  #cv2.putText (img, label, (xmin, ymin - 5), FONT, FONT_SIZE, score, THICKNESS)
  cv2.putText (img, label, (xmin, ymin - SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, label, (xmin, ymin - SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)

def drawScoredRoi (img, roi, label=None, score=None):
  assert score is None or score >= 0 and score <= 1
  if label is None: label = ''
  if score is None:
    score = 0
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3][::-1])
  cv2.rectangle (img, (roi[1], roi[0]), (roi[3], roi[2]), color, THICKNESS)
  #cv2.putText (img, label, (roi[1], roi[0] - SCALE), FONT, FONT_SIZE, score, THICKNESS)
  cv2.putText (img, label, (roi[1], roi[0] - SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, label, (roi[1], roi[0] - SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)


def drawImageId(img, imagefile):
  '''
  Draw the image name in the corner of the image.
  Returns:
    Nothing. Input "img" is changed in place.
  '''
  imheight, imwidth = img.shape[0:2]
  fontscale = float(imheight) / 700
  thickness = min(imheight // 700, 1)
  offsety = imheight // 30
  imageid = op.basename(imagefile)
  cv2.putText (img, imageid, (10, SCALE), FONT, FONT_SIZE, (0,0,0), THICKNESS)
  cv2.putText (img, imageid, (10, SCALE), FONT, FONT_SIZE, (255,255,255), THICKNESS-1)


def drawMaskOnImage(img, mask, alpha=0.5, labelmap=None):
  '''
  Draw a mask on the image, with colors.
  Args:
    img:      numpy array
    mask:     numpy 2-dim array, the same HxW as img.
    labelmap: dict from mask values to output values Key->Value, where:
              Key is scalar int in [0, 255].
              If Value is a scalar, the mask stays grayscale,
              if Value is a tuple (r,g,b), the mask is transformed to color.
              Examples {1: 255, 2: 128} or {1: (255,255,255), 2: (128,255,0)}.
              If not specified, the mask is used as is.
    alpha:    The mask is overlaid with "alpha" transparency.
              alpha=0 means mask is not visible, alpha=1 means img is not visible.
  Returns:
    Output image.
  '''
  if not len(img.shape) == 3:
    raise NotImplementedError('Only color images are supported now.')

  if len(mask.shape) == 3:
    raise NotImplementedError('Color masks are not supported now.')

  if labelmap is not None:
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

