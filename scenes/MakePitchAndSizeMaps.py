''' Creates maps of flattening for each pose for each camera. '''

import argparse
import logging
import numpy as np
from cv2 import resize
from lib.scene import Pose
from lib.homography import getFrameFlattening, getFramePxlsInMeter
import lib.conventions
from lib.iterateScenes import iterateCamerasPoses


def makePitchAndSizeMaps(camera_id, pose_id, dry_run=False):
  ''' Generates maps of pitch and pxls_in_meter for each point in every map. '''

  DEFAULT_HEIGHT = 8.5
  DOWNSCALE = 4  # For speed up and smoothness, compute on downscaled image.

  pose = Pose(camera_id=camera_id, pose_id=pose_id)
  if 'H_pose_to_map' not in pose:
    raise Exception('No homography for camera %d, pose %d' % (camera_id, pose_id))

  H = np.asarray(pose['H_pose_to_map']).reshape((3,3))

  # For each point get a flattening.
  Y = pose.camera['cam_dims']['height']
  X = pose.camera['cam_dims']['width']
  flattening_map = np.zeros((Y // DOWNSCALE, X // DOWNSCALE), dtype=float)
  size_map = np.zeros((Y // DOWNSCALE, X // DOWNSCALE), dtype=float)

  for y in range(Y // DOWNSCALE):
    for x in range(X // DOWNSCALE):
      y_sc = y * DOWNSCALE
      x_sc = x * DOWNSCALE
      flattening_map[y, x] = getFrameFlattening(H, y_sc, x_sc)
      size_map[y, x] = getFramePxlsInMeter(H, pose.map['pxls_in_meter'], y_sc, x_sc)

  logging.info('flattening_map min %.2f, max %.2f' % 
      (np.min(flattening_map), np.max(flattening_map)))
  logging.info('size_map min %.2f, max %.2f' % 
      (np.min(size_map), np.max(size_map)))

  # Top-down is 90 degrees, at the horizon is 0 degrees (consistent with CAD).
  pitch_map = np.arcsin(flattening_map)

  pitch_map = resize((pitch_map * 255.).astype(np.uint8), (X, Y)).astype(float) / 255.
  size_map = resize(size_map.astype(np.uint8), (X, Y)).astype(float)

  pitch_path = lib.conventions.get_pose_pitchmap_path(pose.get_pose_dir())
  size_path = lib.conventions.get_pose_sizemap_path(pose.get_pose_dir())
  
  if not dry_run:
    lib.conventions.write_pitch_image(pitch_path, pitch_map)
    lib.conventions.write_size_image(size_path, size_map)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description='Make pitch and size maps, for one camera-pose or everything.')
  parser.add_argument('--camera_id', type=int, help='if not given, all cameras.')
  parser.add_argument('--pose_id', type=int, help='if not given, all poses.')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  parser.add_argument('--dry_run', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  if args.camera_id is not None and args.pose_id is not None:
    makePitchAndSizeMaps(args.camera_id, args.pose_id, dry_run=args.dry_run)
  elif args.camera_id is None and args.pose_id is None:
    for camera_id, pose_id in iterateCamerasPoses():
      makePitchAndSizeMaps(camera_id, pose_id, dry_run=args.dry_run)
  else:
    raise Exception('Either specify both camera_id and pose_id, or none of them.')
