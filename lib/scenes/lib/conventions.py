import os.path as op
import numpy as np
import cv2


''' File paths conventions. '''

def get_pose_sizemap_path(pose_dir):
  return op.join(pose_dir, 'size-pose.png')

def get_pose_pitchmap_path(pose_dir):
  return op.join(pose_dir, 'pitch-pose.png')

def get_pose_azimuthmap_path(pose_dir):
  # TODO: rename azimuth-frame to azimuth-pose in data folder.
  return op.join(pose_dir, 'azimuth-frame.png')

def get_topdown_azimuthmap_path(pose_dir):
  return op.join(pose_dir, 'azimuth-top-down.png')



''' Read and write the image w.r.t. the conventions. '''


def read_azimuth_image(path):
  ''' Note: returns (None, None) if azimuth map does not exist. '''
  
  if not op.exists(path):
    return None, None
  azimuth = cv2.imread(path, -1)
  assert azimuth is not None
  # Mask is in the alpha channel, the other 3 channels are the same value.
  mask = azimuth[:,:,3] > 0
  azimuth = azimuth[:,:,0].astype(float)
  # By convention, azimuth angles are divided by 2 before written to image.
  azimuth *= 2
  return azimuth, mask


def write_azimuth_image(path, azimuth, mask=None):
  # By convention, azimuth angles are divided by 2 before written to image.
  azimuth = azimuth.copy() / 2.

  # By convention to be human-friendly, write as 8bit.
  azimuth = azimuth.astype(np.uint8)

  assert len(azimuth.shape) == 2, 'Need grayscale azimuth.'
  if mask is None:
    mask = np.ones(azimuth.shape, dtype=np.uint8) * 255
  else:
    assert mask.shape == azimuth.shape
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255

  # Mask is in the alpha channel, the other 3 channels are the same value.
  azimuth = np.stack((azimuth, azimuth, azimuth, mask), axis=-1)
  cv2.imwrite(path, azimuth)


def read_pitch_image(path):
  assert op.exists(path), path
  pitch = cv2.imread(path, -1)
  assert pitch is not None
  assert len(pitch.shape) == 2
  pitch = pitch.astype(float) / 255.
  return pitch

def write_pitch_image(path, pitch):
  pitch = (pitch / (np.pi/2) * 255.).astype(np.uint8)
  cv2.imwrite(path, pitch)


def read_size_image(path):
  ''' No normalization, the range should be [0, 255]. '''
  assert op.exists(path), path
  size = cv2.imread(path, -1)
  assert size is not None
  assert len(size.shape) == 2
  size = size.astype(float)
  return size

def write_size_image(path, size):
  size = size.astype(np.uint8)
  cv2.imwrite(path, size)

