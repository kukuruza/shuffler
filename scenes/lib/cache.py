import os.path as op
import numpy as np
from time import time
import logging
from functools import partial
from .scene import Video
from . import conventions


'''
Store caches of poses, maps, etc. 
The usercase is to go through a db, and get one object per imagefile.
Cache only grows and never cleaned because we have few camera poses.

Extensively uses functions from ./conventions.py
'''


class PoseCache:
  '''
  Store cached poses for an imagefile.
  Internally stores a pose per (camera_id, video_id) tuple.
  '''
  def __init__(self):
    self._cached_poses = {}

  def __getitem__(self, imagefile):
    camera_id, video_id = Video.get_camera_video_ids_from_imagefile(imagefile)

    if (camera_id, video_id) in self._cached_poses:
      pose = self._cached_poses[(camera_id, video_id)]
      logging.debug('PoseCache: took pose from cache for %s' % imagefile)
    else:
      logging.debug('PoseCache: created pose for %s' % imagefile)
      pose = Video(camera_id=camera_id, video_id=video_id).pose
      self._cached_poses[(camera_id, video_id)] = pose
    return pose

  
class MapsCache:
  '''
  Returns cached map [+ mask] for an imagefile.
  Maps are [pose_azimuth, topdown_azimuth, pose_pitch, pose_size].
  Internally stores one per pose.
  '''
  def __init__(self, map_name):
    self._cached_maps = {}
    self._pose_cache = PoseCache()
    def read_map(map_name, pose):
      ''' Factory for different mask names. '''
      if map_name == 'pose_azimuth':
        map_path = conventions.get_pose_azimuthmap_path(pose.get_pose_dir())
        result = conventions.read_azimuth_image(map_path)
      elif map_name == 'topdown_azimuth':
        map_path = conventions.get_topdown_azimuthmap_path(pose.get_pose_dir())
        result = conventions.read_azimuth_image(map_path)
      elif map_name == 'pose_pitch':
        map_path = conventions.get_pose_pitchmap_path(pose.get_pose_dir())
        result = conventions.read_pitch_image(map_path)
      elif map_name == 'pose_size':
        map_path = conventions.get_pose_sizemap_path(pose.get_pose_dir())
        result = conventions.read_size_image(map_path)
      else:
        raise Exception('MapsCache does not recognize map_name %s' % map_name)
      return result
    self._map_name = map_name
    self._read_map_func = partial(read_map, map_name=map_name)

  def __getitem__(self, imagefile):
    pose = self._pose_cache[imagefile]

    if pose in self._cached_maps:
      maps = self._cached_maps[pose]
      logging.debug('MapCache %s: took a map from cache for %s' %
          (self._map_name, imagefile))
    else:
      logging.debug('MapCache: created pose for %s' % imagefile)
      maps = self._read_map_func(pose=pose)
      self._cached_maps[pose] = maps
    return maps
