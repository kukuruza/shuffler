import os, os.path as op
import shutil
import logging
import cv2
import numpy as np
from pprint import pformat
import simplejson as json
from collections import OrderedDict
import matplotlib.pyplot as plt  # for colormaps


def safeCopy (in_path, out_path):
  '''Copy in_path into out_path, which is backed-up if exists.'''

  if not op.exists (in_path):
    raise Exception ('db does not exist: %s' % in_path)
  if op.exists (out_path):
    logging.warning ('will back up existing out_path')
    ext = op.splitext(out_path)[1]
    backup_path = op.splitext(out_path)[0]  + '.backup.%s' % ext
    if in_path != out_path:
      if op.exists (backup_path):
        os.remove (backup_path)
      os.rename (out_path, backup_path)
    else:
      shutil.copyfile(in_path, backup_path)
  if in_path != out_path:
    # copy input database into the output one
    shutil.copyfile(in_path, out_path)

def bbox2roi (bbox):
    assert ((isinstance(bbox, list) or isinstance(bbox, tuple)) and len(bbox) == 4)
    return [bbox[1], bbox[0], bbox[3]+bbox[1]-1, bbox[2]+bbox[0]-1]

def roi2bbox (roi):
    assert ((isinstance(roi, list) or isinstance(roi, tuple)) and len(roi) == 4)
    return [roi[1], roi[0], roi[3]-roi[1]+1, roi[2]-roi[0]+1]

def drawScoredPolygon (img, polygon, label=None, score=None, thickness=2):
  '''
  Args:
    polygon:  a list of coordinates (x,y)
  '''
  assert len(polygon) > 2, polygon
  assert type(polygon[0]) is tuple, polygon
  
  import cv2
  import matplotlib.pyplot as plt  # for colormaps
  font = cv2.FONT_HERSHEY_SIMPLEX
  if label is None: label = ''
  if score is None:
    score = 0
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3])
  for i1 in range(len(polygon)):
    i2 = (i1 + 1) % len(polygon)
    cv2.line(img, polygon[i1], polygon[i2], color, thickness)
  # Draw a label.
  xmin = polygon[0][0]
  ymin = polygon[0][1]
  for i in range(len(polygon)-1):
    xmin = min(xmin, polygon[i][0])
    ymin = min(ymin, polygon[i][1])
  cv2.putText (img, label, (xmin, ymin - 5), font, 0.6, score, thickness)

def drawScoredRoi (img, roi, label=None, score=None, thickness=2):
  assert score is None or score >= 0 and score <= 1
  font = cv2.FONT_HERSHEY_SIMPLEX
  if label is None: label = ''
  if score is None:
    score = 0
  color = tuple([int(x * 255) for x in plt.cm.jet(float(score))][0:3])
  cv2.rectangle (img, (roi[1], roi[0]), (roi[3], roi[2]), color, thickness)
  cv2.putText (img, label, (roi[1], roi[0] - 5), font, 0.6, score, thickness)

def drawFrameId(img, imagefile):
  '''
  Draw the imagename in the corner of the image.
  Returns:
    Nothing. Input "img" is changed in place.
  '''
  font = cv2.FONT_HERSHEY_SIMPLEX
  imheight, imwidth = img.shape[0:2]
  fontscale = float(imheight) / 700
  thickness = imheight / 700
  offsety = imheight / 30
  cv2.putText (img, op.basename(imagefile), (10, 10 + offsety), font, fontscale, (0,0,0), thickness=thickness*3)
  cv2.putText (img, op.basename(imagefile), (10, 10 + offsety), font, fontscale, (255,255,255), thickness=thickness)

def drawMaskOnImage(img, mask, labelmap, alpha=0.5):
  '''
  Draw a mask on the image, with colors.
  Args:
    img: numpy array
    mask: numpy 2-dim array, the same size as img.
    labelmap: an ordered dict, see the function loadLabelmap.
  Returns:
    Output image.
  '''
  labels = np.unique(mask).tolist()
  logging.debug('drawMaskOnImage: found labels: %s' % str(labels))
  maskR = np.zeros(mask.shape, dtype=np.uint8)
  maskG = np.zeros(mask.shape, dtype=np.uint8)
  maskB = np.zeros(mask.shape, dtype=np.uint8)
  logging.debug('drawMaskOnImage: labelmap: %s' % labelmap)
  for label in labels:
    labelRGBA = labelmap[label]['color'] if label in labelmap else [0,0,0]
    logging.debug('drawMaskOnImage: label: %d, color: %s, numpixels: %d' %
        (label, labelRGBA, np.count_nonzero(mask == label)))
    maskR[mask == label] = labelRGBA[0]
    maskG[mask == label] = labelRGBA[1]
    maskB[mask == label] = labelRGBA[2]
  logging.debug('drawMaskOnImage: alpha %.3f' % alpha)
  maskRGB = np.stack([maskR, maskG, maskB], axis=2)
  if maskRGB.shape != img.shape:
    maskRGB = cv2.resize(maskRGB, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
  img = cv2.addWeighted(src1=img, alpha=1, src2=maskRGB, beta=alpha, gamma=0)
  return img

def loadLabelmap(labelmap_path):
  ''' Load the labelmap file w.r.t the conventions. '''

  logging.info('loadLabelmap: loading %s' % labelmap_path)
  if not op.exists(labelmap_path):
    raise Exception('Labelmap file does not exist: %s' % labelmap_path)
  with open(labelmap_path) as f:
    labelmap = json.load(f)
    labelmap = labelmap['labelmap']
    logging.info(pformat(labelmap))

  # Create an ordered dict out of the list.
  labelmap = OrderedDict([(int(key), value) for key, value in labelmap.iteritems()])
  return labelmap
  
  