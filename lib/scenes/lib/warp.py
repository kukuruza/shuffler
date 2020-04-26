#!/usr/bin/env python
import sys, os, os.path as op
from scenes.lib.scene import Pose
import simplejson as json
import logging
import argparse
import cv2
import numpy as np
from pprint import pprint
from .scene import Pose, Video


def _intMaskedDilation(img, kernel):
  ''' Zero neighbours of a pixel do not count in the result for that pixel. '''

  dtype = img.dtype  # Remember the input type for output.
  mask = (img > 0).astype(np.uint16)
  img = img.astype(np.uint16)

  assert len(kernel.shape) == 2, kernel.shape
  img = cv2.filter2D(img, -1, kernel)
  mask = cv2.filter2D(mask, -1, kernel)
  with np.errstate(divide='ignore', invalid='ignore'):
    dilated = np.true_divide( img.astype(float), mask.astype(float) )
    dilated[ ~ np.isfinite( dilated )] = 0  # -inf inf NaN
  return dilated.astype(dtype)


def transformPoint(H, x, y):
  ''' Apply homography to a point. '''

  p = np.asarray([[x],[y],[1.]])
  p = np.matmul(H, p)
  if p[2] == 0:
    return float('Inf'), float('Inf')
  else:
    p /= p[2]
    x, y = p[0,0], p[1,0]
    return x, y


def warp(in_image, H, dims_in, dims_out,
    dilation_radius=None, no_alpha=False):
  ''' Warp image-to-image using provided homograhpy.
  Args:
    dims_in:   None or (H, W) homography input dimensions.
               If given, image will be resized to it.
    dims_out:  (H, W) homography output dimensions.
    H:         3x3 numpy array; to invert the direction, use np.linalg.inv(H)
  Returns:
    wrapped image of dimensions 'dims_out'.
  '''

  if dims_in is None:
    dims_in = in_image.shape[0:2]

  np.set_printoptions(precision=1, formatter = dict( float = lambda x: "%10.4f" % x ))
  logging.debug (str(H))
  logging.debug ('dims_in: %s' % str(dims_in))
  logging.debug ('dims_out: %s' % str(dims_out))

  assert in_image is not None
  logging.debug('Type of input image: %s' % str(in_image.dtype))

  if dilation_radius is not None and dilation_radius > 0:
    kernelsize = dilation_radius * 2 + 1
    kernel = np.ones((kernelsize, kernelsize), dtype=int)
    in_image = _intMaskedDilation(in_image, kernel)

  # If input is on the wrong scale, need to use another homography first.
  in_scale_h = float(dims_in[0]) / in_image.shape[0]
  in_scale_w = float(dims_in[1]) / in_image.shape[1]
  assert (in_scale_w - in_scale_h) / (in_scale_w + in_scale_h) < 0.01, (dims_in, in_image.shape)
  H_scale = np.identity(3, dtype=float)
  H_scale[0,0] = in_scale_h
  H_scale[1,1] = in_scale_w
  H_scale[2,2] = 1.
  H = np.matmul(H, H_scale)
  logging.debug('Scaling H with (%f %f)' % (in_scale_h, in_scale_w))
  logging.debug('Using H to warp: %s' % str(H))

  out_image = cv2.warpPerspective(in_image, H, (dims_out[1], dims_out[0]), flags=cv2.INTER_NEAREST)
  logging.debug('Type of output image: %s' % str(out_image.dtype))

  # Remove alpha, if necessary.
  if len(out_image.shape) == 3 and out_image.shape[2] == 4 and no_alpha:
    out_image[:,:,0][np.bitwise_not(out_image[:,:,3])] = 0
    out_image[:,:,1][np.bitwise_not(out_image[:,:,3])] = 0
    out_image[:,:,2][np.bitwise_not(out_image[:,:,3])] = 0
    out_image = out_image[:,:,0:3]

  return out_image


def warpPoseToMap(in_image, camera_id, pose_id,
                  reverse_direction=False, dilation_radius=None, no_alpha=False):
  ''' Warp from Pose to Map or vice-versa. '''

  pose = Pose(camera_id, pose_id=pose_id)
  H = np.asarray(pose['H_pose_to_map']).reshape((3,3))
  dims_in = (pose.camera['cam_dims']['height'], pose.camera['cam_dims']['width'])
  dims_out = (pose.map['map_dims']['height'], pose.map['map_dims']['width'])
  if reverse_direction:
    H = np.linalg.inv(H)
    dims_in, dims_out = dims_out, dims_in
  return warp(in_image, H, dims_in, dims_out,
              dilation_radius=dilation_radius, no_alpha=no_alpha)


def warpVideoToMap(in_image, camera_id, video_id,
                  reverse_direction=False, dilation_radius=None, no_alpha=False):
  ''' Warp from Video to Map and vice-versa. '''

  video = Video(camera_id=camera_id, video_id=video_id)
  if 'H_video_to_pose' in video:
    H_video_to_pose = np.asarray(video['H_video_to_pose']).reshape((3,3))
  else:
    H_video_to_pose = np.eye(3, dtype=float)
  H_pose_to_map = np.asarray(video.pose['H_pose_to_map']).reshape((3,3))
  H = np.matmul(H_pose_to_map, H_video_to_pose)
  # TODO: for dims_in sometimes video and camera dims differ.
  dims_in = (video.pose.camera['cam_dims']['height'], video.pose.camera['cam_dims']['width'])
  dims_out = (video.pose.map['map_dims']['height'], video.pose.map['map_dims']['width'])
  if reverse_direction:
    H = np.linalg.inv(H)
    dims_in, dims_out = dims_out, dims_in
  return warp(in_image, H, dims_in, dims_out,
              dilation_radius=dilation_radius, no_alpha=no_alpha)


def warpVideoToPose(in_image, camera_id, video_id,
                    reverse_direction=False, dilation_radius=None, no_alpha=False):
  ''' Warp from Video to Map and vice-versa. '''

  video = Video(camera_id=camera_id, video_id=video_id)
  if 'H_video_to_pose' in video:
    H = np.asarray(video['H_video_to_pose']).reshape((3,3))
  else:
    H = np.eye(3, dtype=float)
  dims_out = (video.pose.camera['cam_dims']['height'], video.pose.camera['cam_dims']['width'])
  dims_in = dims_out  # TODO: for dims_in sometimes video and camera dims differ.
  if reverse_direction:
    H = np.linalg.inv(H)
    dims_in, dims_out = dims_out, dims_in
  return warp(in_image, H, dims_in, dims_out,
              dilation_radius=dilation_radius, no_alpha=no_alpha)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Demonstration of warpFrameToMap.')
  parser.add_argument('--camera', required=True, help="E.g., '572'.")
  parser.add_argument('--in_image_path', required=True)
  parser.add_argument('--out_image_path')
  parser.add_argument('--pose_id', type=int, default=0)
  parser.add_argument('--dilation_radius', help='Optional dilation on input image.', type=int)
  parser.add_argument('--reverse_direction', action='store_true')
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Using cv2.imread instead of scipy.misc because cv2 correctly processes 16bits.
  assert op.exists(args.in_image_path), args.in_image_path
  in_image = cv2.imread(args.in_image_path, -1)

  out_image = warpFrameToMap(in_image, args.camera, args.pose_id,
      dilation_radius=args.dilation_radius,
      reverse_direction=args.reverse_direction)

  if args.out_image_path:
    cv2.imwrite(args.out_image_path, out_image)

