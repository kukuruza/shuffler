#!/usr/bin/env python
import os, os.path as op
from lib.scene import Camera, Pose
import logging
import argparse
import cv2
import numpy as np
import math
from imageio import imwrite, get_writer
from lib.scene import Pose
from lib.labelMatches import loadMatches, getGifFrame
from lib.warp import warp


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--camera_id', required=True, type=int)
  parser.add_argument('--map_id', type=int, help='If not set, will use pose["best_map_id"]')
  parser.add_argument('--pose_id', type=int, default=0)
  parser.add_argument('--height', type=float, default=8.5)
  parser.add_argument('--no_backup', action='store_true')
  parser.add_argument('--update_map_json', action='store_true')
  parser.add_argument('--ransac', action='store_true')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  pose = Pose(args.camera_id, pose_id=args.pose_id)

  # Load matches.
  matches_path = op.join(pose.get_pose_dir(), 'matches-map%d.json' % pose.map_id)
  assert op.exists(matches_path), matches_path
  src_pts, dst_pts = loadMatches(matches_path, 'pose', 'map')

  # Compute video->pose homography.
  src_pts = src_pts.reshape(-1,1,2)
  dst_pts = dst_pts.reshape(-1,1,2)
  method = cv2.RANSAC if args.ransac else 0
  H, _ = cv2.findHomography(src_pts, dst_pts, method=method)
  pose['H_pose_to_map'] = H.copy().reshape((-1,)).tolist()
  print H

  mapH = pose.map['map_dims']['height']
  mapW = pose.map['map_dims']['width']
  frameH = pose.camera['cam_dims']['height']
  frameW = pose.camera['cam_dims']['width']

  # Compute the origin on the map.
  satellite = pose.map.load_satellite()
  X1 = H[:,1].copy().reshape(-1)
  X1 /= X1[2]
  logging.debug('Vertically down line projected onto map: %s' % str(X1))
  X2 = np.dot(H, np.asarray([[frameW/2],[frameH/2],[1.]])).reshape(-1)
  X2 /= X2[2]
  logging.debug('Frame center projected onto map: %s' % str(X2))
  cv2.circle(satellite, (int(X2[0]),int(X2[1])), 6, (0,255,0), 4)
  if 'cam_origin' in pose.camera and 'z' in pose.camera['cam_origin']:
    # TODO: find out how to infer height, focal length, and FOV from homography.
    h = pose.camera['cam_origin']['z']
  else:
    logging.warning('No camera height, will use given in args')
    h = args.height
  h *= pose.map['pxls_in_meter']
  l = math.sqrt((X1[0]-X2[0])*(X1[0]-X2[0])+(X1[1]-X2[1])*(X1[1]-X2[1]))
  d = math.sqrt(1 - 4*h*h/l/l)
  logging.info ('Discriminant for origin computation: %.2f' % d)
  # Assume the camera is not looking that much down.
  X0 = (1. + d) / 2 * X1 + (1. - d) / 2 * X2
  logging.info('Camera origin (x,y) on the map: (%d,%d)' % (X0[0], X0[1]))
  cv2.circle(satellite, (int(X0[0]),int(X0[1])), 6, (0,255,0), 4)

  # Save. Propagate this infomation forward on to map.json.
  if args.update_map_json:
    pose.map['map_origin'] = {
        'x': int(X0[0]), 'y': int(X0[1]), 
        'comment': 'computed by ComputeH from pose%d' % pose.pose_id
    }
  pose.save(backup=not args.no_backup)

  # Warp satellite for nice visualization.
  warped_satellite = warp(satellite, np.linalg.inv(H), (mapH, mapW), (frameH, frameW))
  warped_path = op.join(pose.get_pose_dir(), 'satellite-warped-map%d.gif' % pose.map_id)
  poseframe = pose.load_example()
  with get_writer(warped_path, mode='I') as writer:
    N = 15
    for i in range(N):
      writer.append_data(getGifFrame(warped_satellite, poseframe, float(i) / N))

  # Make visibility map.
  # Horizon line.
  horizon = H[2,:].copy().transpose()
  logging.debug('Horizon line: %s' % str(horizon))
  assert horizon[1] != 0
  x1 = 0
  x2 = frameW-1
  y1 = int(- (horizon[0] * x1 + horizon[2]) / horizon[1])
  y2 = int(- (horizon[0] * x2 + horizon[2]) / horizon[1])
  # Visible part in the frame.
  visibleframe = np.ones((frameH, frameW), np.uint8) * 255
  cv2.fillPoly(visibleframe, np.asarray([[(x1,y1),(x2,y2),(x2,0),(x1,0)]]), (0,))
  # Visible part in the satallite.
  visiblemap = warp(visibleframe, H, (frameH, frameW), (mapH, mapW))
  # Would be nice to store visiblemap as alpha channel, but png takes too much space.
  alpha_mult = np.tile(visiblemap.astype(float)[:,:,np.newaxis] / 512 + 0.5, 3)
  visiblemap = (satellite * alpha_mult).astype(np.uint8)
  visibility_path = op.join(pose.get_pose_dir(), 'visible-map%d.jpg' % pose.map_id)
  imwrite(visibility_path, visiblemap)
