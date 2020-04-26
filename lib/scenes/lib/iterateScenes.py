from glob import glob
import simplejson as json
import os.path as op
import re


def iterateCamerasPoses():
  ''' Go through all valid (camera_id, pose_id). '''

  cam_dirs = sorted(glob('data/scenes/???'))
  for cam_dir in cam_dirs:

    camera_id = int(op.basename(cam_dir))
    cam_info = json.load(open(op.join(cam_dir, 'camera.json')))
    cam_info['poses'] = []

    for pose_dir in sorted(glob(op.join(cam_dir, 'pose*'))):
      pose_name = op.basename(pose_dir)
      pose_id = int(re.findall(r'\d+', pose_name)[0])
      yield camera_id, pose_id
