import os.path as op
import numpy as np
from time import time
import logging
from .scene import Video
from .conventions import read_azimuth_image



def getHfromPose(pose):
  assert 'H_pose_to_map' in pose, pose
  H = np.asarray(pose['H_pose_to_map']).reshape((3,3))
  logging.debug('H: %s' % str(H))
  return H


def transformPoint():
  assert 0, 'The function moved to warp.py.'


def getFrameFlattening(H, y_frame, x_frame):
  ''' Compute flattening == sin(pitch) for a point in a frame. '''

  def _getMapEllipse(H, y_frame, x_frame):
    assert H is not None
    p_frame = np.asarray([[x_frame],[y_frame],[1.]])
    p_frame_dx = np.asarray([[x_frame + 1.],[y_frame],[1.]])
    p_frame_dy = np.asarray([[x_frame],[y_frame + 1.],[1.]])
    p_map = np.matmul(H, p_frame)
    p_map /= p_map[2]
    p_map_dx = np.matmul(H, p_frame_dx)
    p_map_dx /= p_map_dx[2]
    p_map_dy = np.matmul(H, p_frame_dy)
    p_map_dy /= p_map_dy[2]
    return p_map_dx - p_map, p_map_dy - p_map

  assert H is not None
  dx, dy = _getMapEllipse(H, y_frame, x_frame)
  flattening = np.linalg.norm(dx, ord=2) / np.linalg.norm(dy, ord=2)
  #logging.debug('Flattening: %.2f' % flattening)
  return flattening


def getFramePxlsInMeter(H, pxls_in_meter, y_frame, x_frame):
  ''' Compute size (in meters) for a point in a frame. '''

  def _getMapDxMeters(H, y_frame, x_frame):
    assert H is not None
    p_frame = np.asarray([[x_frame],[y_frame],[1.]])
    p_frame_dx = np.asarray([[x_frame + 1.],[y_frame],[1.]])
    p_map = np.matmul(H, p_frame)
    p_map /= p_map[2]
    p_map_dx = np.matmul(H, p_frame_dx)
    p_map_dx /= p_map_dx[2]
    return p_map_dx - p_map

  assert H is not None
  dx_pxl = _getMapDxMeters(H, y_frame, x_frame)
  dx_meters = pxls_in_meter / np.linalg.norm(dx_pxl, ord=2)
  #logging.debug('Size, meters: %.2f' % dx_meters)
  return dx_meters

