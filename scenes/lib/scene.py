import os, os.path as op
import shutil
import logging
import simplejson as json
from cv2 import imread
from glob import glob
import re
import numpy as np


def _atscenes(path):
  return op.join(os.getenv('SCENES_PATH'), path)


class Info:
  def __init__(self):
    self.info = None
    self.path = None

  def __getitem__(self, key):
    assert key in self.info, self.dump()
    return self.info[key]

  def __setitem__(self, key, item):
    self.info[key] = item

  def __contains__(self, key):
    return True if key in self.info else False

  def dump(self):
    mystr = json.dumps(self.info, sort_keys=True, indent=2)
    return mystr

  def load(self):
    assert op.exists(self.path), self.path
    logging.debug ('%s: loading info from: %s' % (self.__class__.__name__, self.path))
    self.info = json.load(open(self.path))
    logging.debug(self.dump())

  def save(self, backup=True):
    if backup:
      backup_path = op.splitext(self.path)[0] + '.backup.json'
      if op.exists(self.path):
        shutil.copyfile(self.path, backup_path)
    with open(self.path, 'w') as outfile:
      outfile.write(self.dump())


class Camera(Info):
  def __init__ (self, camera_id):
    Info.__init__(self)
    self.camera_id = camera_id
    self.path = _atscenes(op.join('%s' % camera_id, 'camera.json'))
    self.load()

  def get_camera_dir(self):
    return op.dirname(self.path)


class Map(Info):
  def __init__(self, camera_id, map_id):
    Info.__init__(self)
    self.camera_id = camera_id
    self.map_id = map_id
    self.path = op.join(self.get_map_dir(), 'map.json')
    self.load()

  def get_map_dir(self):
    return _atscenes(op.join('%s' % self.camera_id, 'map%d' % self.map_id))

  def load_satellite(self):
    satellite_path = op.join(self.get_map_dir(), 'satellite.jpg')
    assert op.exists(satellite_path), satellite_path
    satellite = imread(satellite_path)
    assert satellite.shape[:2] == (self['map_dims']['height'], self['map_dims']['width'])
    return satellite


class Pose(Info):
  def __init__(self, camera_id, pose_id):
    Info.__init__(self)
    self.camera_id = camera_id
    self.pose_id = pose_id
    self.camera = Camera(self.camera_id)
    # First camera, then can do self.get_pose_dir() and then self.load().
    self.path = op.join(self.get_pose_dir(), 'pose.json')
    self.load()
    # First load, then can get self['map_id']
    self.map_id = self['map_id'] if 'map_id' in self else 0
    self.map = Map(self.camera_id, self.map_id)
    logging.info('Pose: loaded cam %s, map %d, pose %d.' %
        (self.camera_id, self.map_id, self.pose_id))

  def get_pose_dir(self):
    return _atscenes(op.join('%s' % self.camera_id, 'pose%d' % self.pose_id))

  def save(self, backup=True):
    Info.save(self, backup=backup)
    self.camera.save(backup=backup)
    self.map.save(backup=backup)

  def load_example(self):
    frame_pattern = op.join(self.get_pose_dir(), 'example*.*')
    frame_paths = glob(frame_pattern)
    assert len(frame_paths) > 0, frame_pattern
    frame_path = frame_paths[0]  # Take a single frame.
    frame = imread(frame_path)
    assert frame.shape[:2] == (
      self.camera['cam_dims']['height'], self.camera['cam_dims']['width'])
    return frame


class Video(Info):
  def __init__ (self, camera_id, video_id):
    Info.__init__(self)
    self.camera_id = camera_id
    self.video_id = video_id
    self.path = op.join(self.get_video_dir(), 'video.json')
    self.load()
    self.pose = Pose(camera_id=self.camera_id, pose_id=self['pose_id'])

  def get_video_dir(self):
    return _atscenes(op.join('%s' % self.camera_id, 'videos', self.video_id))

  def load (self):
    multiple_videos_path = op.abspath(op.join(self.get_video_dir(), '../videos.json'))
    # Video-specific json.
    if op.exists(self.path):
      logging.info ('Video: loading videos info from: %s' % self.path)
      Info.load(self)
      assert 'pose_id' in self, '%s does not have pose_id.' % self.path
    # Multiple-videos json file.
    elif op.exists(multiple_videos_path):
      logging.debug('Video: %s does not exist.' % self.path)
      logging.info ('Video: loading videos info from: %s' % multiple_videos_path)
      videos = json.load(open(multiple_videos_path))
      assert 'videos' in videos, json.dumps(videos, sort_keys=True, indent=2)
      videos = videos['videos']
      if self.video_id not in videos:
        raise Exception('Camera has videos.json, but video %s is not there:\n%s' %
            (self.video_id, json.dumps(videos, sort_keys=True, indent=2)))
      # Nothing besides pose_id is expected in videos.json file.
      assert 'pose_id' in videos[self.video_id]
      self.info = {'pose_id': videos[self.video_id]['pose_id']}
    # No video info.
    else:
      logging.info('Video: %s or %s not found.' % (multiple_videos_path, self.path))
      self.info = {'pose_id': 0}

  def save(self, backup=True):
    Info.save(self, backup=backup)
    self.pose.save(backup=backup)

  def load_example(self):
    frame_pattern = op.join(self.get_video_dir(), 'example*.*')
    frame_paths = glob(frame_pattern)
    assert len(frame_paths) > 0, frame_pattern
    frame_path = frame_paths[0]  # Take a single frame.
    frame = imread(frame_path)
    return frame

  @classmethod
  def from_imagefile(cls, imagefile):
    ''' Parse camera_id and video_id from the imagefile, and create a pose. '''

    camera_id, video_id = Video.get_camera_video_ids_from_imagefile(imagefile)
    return Video(camera_id, video_id)

  @classmethod
  def get_camera_video_ids_from_imagefile(cls, imagefile):
    ''' Parse camera_id and video_id from the imagefile,
    to be used by Video constructor and caches, which do not need Video object. '''

    dirs = imagefile.split('/')
    # Find camera_id.
    matches_cam = [re.compile('^(cam(\d{3})|\d{3})$').match(x) for x in dirs]
    try:
      cam_ind = next(i for i, j in enumerate(matches_cam) if j)  # Find first not None
      camera_name = matches_cam[cam_ind].groups()[0]
    except StopIteration:
      logging.info('Failed to deduce camera from %s' % imagefile)
      return None
    logging.debug('Deduced camera_name to be %s' % camera_name)
    # Gradually moving to remove 'cam' prefix from video files.
    camera_id = re.search('\d+', camera_name).group(0)
    logging.debug('Deduced camera_id to be %s' % camera_id)
    # Find video_id.
    assert cam_ind + 1 < len(dirs), 'Camera should not be the last in %s' % imagefile
    video_id = dirs[cam_ind+1]  # The next dir after camera.
    video_id = op.splitext(video_id)[0]  # In case .avi file is given.
    return camera_id, video_id




if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

  # Low level construction of camera, map and pose objects.
  camera = Camera(camera_id=170)
  map = Map(camera_id=170, map_id=0)
  pose = Pose(camera_id='170', pose_id=1)
  video = Video(camera_id=181, video_id='181-20151224-15')
  video = Video(camera_id=170, video_id='170-20160502-18')

  # Example of construction Pose by passing imagefile as in databases.
  video = Video.from_imagefile('data/camdata/170/170-20160508-18/000002')
