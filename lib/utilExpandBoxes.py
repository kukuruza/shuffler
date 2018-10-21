import logging

from .util import roi2bbox

def expandRoiBorder (roi, imsize, perc, integer_result=True):
  '''Expands a ROI, and clips it within borders.
  Floats are rounded to the nearest integer.
  '''
  imheight, imwidth = imsize
  perc_y, perc_x = perc
  if (perc_y, perc_x) == (0, 0): return roi

  half_delta_y = float(roi[2] + 1 - roi[0]) * perc_y / 2
  half_delta_x = float(roi[3] + 1 - roi[1]) * perc_x / 2
  # the result must be within (imheight, imwidth)
  bbox_height = roi[2] + 1 - roi[0] + half_delta_y * 2
  bbox_width  = roi[3] + 1 - roi[1] + half_delta_x * 2
  if bbox_height > imheight or bbox_width > imwidth:
    logging.warning ('expanded bbox of size (%d,%d) does not fit into image (%d,%d)' %
        (bbox_height, bbox_width, imheight, imwidth))
    # if so, decrease half_delta_y, half_delta_x
    coef = min(imheight / bbox_height, imwidth / bbox_width)
    logging.warning ('decreased bbox to (%d,%d)' % (bbox_height, bbox_width))
    bbox_height *= coef
    bbox_width  *= coef
    #logging.warning ('imheight, imwidth (%d,%d)' % (imheight, imwidth))
    logging.warning ('decreased bbox to (%d,%d)' % (bbox_height, bbox_width))
    half_delta_y = (bbox_height - (roi[2] + 1 - roi[0])) * 0.5
    half_delta_x = (bbox_width  - (roi[3] + 1 - roi[1])) * 0.5
    #logging.warning ('perc_y, perc_x: %.1f, %1.f: ' % (perc_y, perc_x))
    #logging.warning ('original roi: %s' % str(roi))
    #logging.warning ('half_delta-s y: %.1f, x: %.1f' % (half_delta_y, half_delta_x))
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
  if roi[2] > imheight-1:
    roi[0] -= abs((imheight-1) - roi[2])
    roi[2] = imheight-1
  if roi[3] > imwidth-1:
    roi[1] -= abs((imwidth-1) - roi[3])
    roi[3] = imwidth-1
  # check that now averything is within borders (bbox is not too big)
  assert roi[0] >= 0 and roi[1] >= 0, str(roi)
  assert roi[2] <= imheight-1 and roi[3] <= imwidth-1, str(roi)
  # make integer
  if integer_result:
    roi = [int(round(x)) for x in roi]
  return roi


def expandRoi (roi, perc, integer_result=True):
  '''Expands a ROI. Floats are rounded to the nearest integer.
  '''
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


def expandRoiToRatioBorder (roi, imsize, expand_perc, ratio):
  '''Expands a ROI to keep 'ratio', and maybe more, up to 'expand_perc'
  '''
  imheight, imwidth = imsize
  bbox = roi2bbox(roi)
  # adjust width and height to ratio
  height = float(roi[2] + 1 - roi[0])
  width  = float(roi[3] + 1 - roi[1])
  if height / width < ratio:
    perc = ratio * width / height - 1
    roi = expandRoiBorder (roi, (imheight, imwidth), (perc, 0), integer_result=False)
  else:
    perc = height / width / ratio - 1
    roi = expandRoiBorder (roi, (imheight, imwidth), (0, perc), integer_result=False)
  # additional expansion
  perc = (1 + expand_perc) / (1 + perc) - 1
  if perc > 0:
    roi = expandRoi (roi, (imheight, imwidth), (perc, perc), integer_result=False)
  roi = [int(round(x)) for x in roi]
  return roi


def expandRoiToRatio (roi, expand_perc, ratio):
  '''Expands a ROI to keep 'ratio', and maybe more, up to 'expand_perc'
  '''
  bbox = roi2bbox(roi)
  # adjust width and height to ratio
  height = float(roi[2] + 1 - roi[0])
  width  = float(roi[3] + 1 - roi[1])
  if height / width < ratio:
   perc = ratio * width / height - 1
   roi = expandRoi (roi, (perc, 0), integer_result=False)
  else:
   perc = height / width / ratio - 1
   roi = expandRoi (roi, (0, perc), integer_result=False)
  # additional expansion
  #perc = (1 + expand_perc) / (1 + perc) - 1
  #if perc > 0:
  roi = expandRoi (roi, (expand_perc, expand_perc))
  roi = [int(round(x)) for x in roi]
  return roi

