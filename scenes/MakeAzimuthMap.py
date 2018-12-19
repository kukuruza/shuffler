import os, os.path as op
import numpy as np
import logging
from glob import glob
import argparse
import cv2
import subprocess
from skimage.measure import label
from lib.warp import warpPoseToMap
from lib.scene import Pose
from lib.conventions import read_azimuth_image, write_azimuth_image


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=
      '''The pipeline for computing azimuth maps.
      All azimuth are ''')
  parser.add_argument('--camera_id', required=True, help="E.g. 572")
  parser.add_argument('--pose_id', type=int, default=0)
  parser.add_argument('--steps', nargs='+',
      choices=['2topdown', 'compute_azimuth', 'delta', '2frame'])
  parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  pose = Pose(camera_id=args.camera_id, pose_id=args.pose_id)

  pose_dir = pose.get_pose_dir()
  for_azimuth_dir = op.join(pose_dir, '4azimuth-map%d' % pose.map_id)
  logging.info ('Looking at directory: %s' % for_azimuth_dir)
  if not op.exists(for_azimuth_dir):
    os.makedirs(for_azimuth_dir)

  if args.steps is None or '2topdown' in args.steps:
    lane_paths = glob(op.join(for_azimuth_dir, 'lanes*16b.png'))
    print ('Found %d lane images' % len(lane_paths))

    for lane_path in lane_paths:
      print (lane_path)
      assert op.exists(lane_path), lane_path
      in_lane = cv2.imread(lane_path, -1)[:,:,0]
#      assert in_lane.dtype == np.uint16, 'Need 16 bits input image.'

      # # Find the min radius that does not ruin lanes.
      # counts = []
      # for dilation_radius in range(5):
      #   warped_lane = warpPoseToMap(in_lane, args.camera_id, args.pose_id,
      #       dilation_radius=dilation_radius, no_alpha=True)
      #   _, count = label((warped_lane > 0).astype(int), connectivity=2, return_num=True)
      #   counts.append(count)
      # print ('Lanes segments counts for different dilation', counts)
      # first_index_of_min_count = counts.index(min(counts))
  
      # Do again with the optimal radius, and write to disk.
      dilation_radius = 2 # range(5)[first_index_of_min_count]
      warped_lane = warpPoseToMap(in_lane, args.camera_id, args.pose_id,
          dilation_radius=dilation_radius, no_alpha=True)
      _, count = label((warped_lane > 0).astype(int), connectivity=1, return_num=True)
      print ('Used dilation radius %d, got %d segments.' %
          (dilation_radius, count))
      warped_lane_path = op.splitext(lane_path)[0] + '-warped.png'
      cv2.imwrite(warped_lane_path, warped_lane)

  # Calculate azimuth on the top-down view.
  if args.steps is None or 'compute_azimuth' in args.steps:
    rel_pose_dir = op.relpath(pose_dir, os.getenv('CITY_PATH'))
    rel_azimuth_dir = op.relpath(for_azimuth_dir, os.getenv('CITY_PATH'))
    s = ('''"cd(fullfile(getenv('CITY_PATH'), 'src/scenes/lib')); ''' +
         '''ComputeAzimuthMap('%s/lanes*warped.png', '%s/azimuth-top-down.png'); '''
         % (rel_azimuth_dir, rel_pose_dir) + '''exit;" ''' )
    command = '%s/bin/matlab' % os.getenv('MATLAB_HOME') + ' -nodisplay -r ' + s
    logging.info (command)
    returncode = subprocess.call (command, shell=True, executable="/bin/bash")

  if args.steps is None or 'delta' in args.steps:
    print ('Creating delta azimuth for top-down view.')
    azimuth_top_down_path = op.join(pose_dir, 'azimuth-top-down.png')
    azimuth_top_down, mask_top_down = read_azimuth_image(azimuth_top_down_path)
   
    # Camera can see ech point at a certain angle.
    origin = pose.map['map_origin']
    X, Y = azimuth_top_down.shape[1], azimuth_top_down.shape[0]
    delta_x_1D = np.arange(X) - origin['x']
    delta_y_1D = np.arange(Y) - origin['y']
    delta_x = np.dot( np.ones((Y,1)), delta_x_1D[np.newaxis,:] ).astype(float)
    delta_y = np.dot( delta_y_1D[:,np.newaxis], np.ones((1,X)) ).astype(float)
    delta_azimuth = - np.arctan2 (delta_x, -delta_y)  # 0 is north, 90 is west.
    # From [-pi, pi] to [0 360]
    delta_azimuth = np.mod( (delta_azimuth * 180. / np.pi), 360. )
    # Write for debugging.
    delta_azimuth_path = op.join(for_azimuth_dir, 'azimuth-top-down-delta.png')
    write_azimuth_image(delta_azimuth_path, delta_azimuth)
    # Top-down azimuth in camera point of view, 0 is north, 90 is west.
    azimuth_top_down = np.mod(azimuth_top_down - delta_azimuth, 360.)
    azimuth_top_down_path = op.join(for_azimuth_dir, 'azimuth-top-down-from-camera.png')
    write_azimuth_image(azimuth_top_down_path, azimuth_top_down, mask_top_down)

  if args.steps is None or '2frame' in args.steps:
    print ('Warping from top-down view to frame view.')
    azimuth_top_down_path = op.join(for_azimuth_dir, 'azimuth-top-down-from-camera.png')
    azimuth_top_down, mask_top_down = read_azimuth_image(azimuth_top_down_path)
    azimuth_top_down[mask_top_down == 0] = 0
    azimuth_frame = warpPoseToMap(azimuth_top_down, 
        args.camera_id, args.pose_id,
        dilation_radius=2, reverse_direction=True)
    mask_frame = warpPoseToMap(mask_top_down.astype(float),
        args.camera_id, args.pose_id,
        dilation_radius=2, reverse_direction=True) > 0.5
    azimuth_frame_path = op.join(pose_dir, 'azimuth-frame.png')
    write_azimuth_image(azimuth_frame_path, azimuth_frame, mask_frame)
