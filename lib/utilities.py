import os, os.path as op
import shutil
import logging
import cv2
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

def relabelMask(mask, labelmap):
  assert len(labelmap) > 0
  firstval = labelmap.itervalues().next()
  if len(firstval) == 1:
    # Grayscale.
    out = np.zeros(shape=mask.shape, dtype=np.uint8)
    for key, value in sorted(labelmap.items()):
      key = int(key)  # To be able to store it as json.
      assert len(value) == 1, 'All value should have len 1: %s' % value
      out[mask[key]] = value
  elif len(firstval) == 3 or len(firstval) == 4:
    # RGB or RGBA.
    out = np.zeros(shape=[mask.shape[0], mask.shape[1], len(firstval)], dtype=np.uint8)
    for key, value in sorted(labelmap.items()):
      key = int(key)  # To be able to store it as json.
      assert len(value) == len(firstval), 'All value should have len %d: %s' % (len(firstval), value)
      out[mask[key],:] = value
  else:
    raise ValueError('Can only map to 1, 3, or 4 channels.')
  return out