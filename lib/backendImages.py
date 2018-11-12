import os, sys, os.path as op
import numpy as np
import imageio
import logging
import struct
import cv2
import traceback
from pprint import pformat
from PIL import Image


'''
The support for reading and writing media in the form of single files and
multiple files, which are supported by our backend imageio:
https://imageio.readthedocs.io/en/stable/formats.html

Levels of abstraction:

- classes VideoReader and PictureReader:
  Hide 1) implementation if "imread" for images and "maskread" for masks,
  2) maintain cache of images already in memory,
  3) maintain a collection of open videos (in case of VideoReader),
  4) compute absolute paths from "rootdir" and the provided paths.

- class ImageryReader:
  Hides 1) whether VideoReader or PictureReader should be used.
'''


def getPictureSize(imagepath):
  if not op.exists (imagepath):
    raise ValueError('Image does not exist at path: "%s"' % imagepath)
  im = Image.open(imagepath)
  width, height = im.size
  return height, width


class VideoReader:
  '''Implementation of imagery reader based on "Image" <-> "Frame in video".'''

  def __init__ (self, rootdir):
    '''
    Args:
      rootdir:     imagefile and maskfile are considered to be  paths relative to rootdir.
    '''
    self.rootdir = rootdir
    logging.info('Root is set to %s' % self.rootdir)
    self.image_cache = {}    # cache of previously read image(s)
    self.mask_cache = {}     # cache of previously read mask(s)
    self.image_video = {}    # map from image video name to VideoCapture object
    self.mask_video = {}     # map from mask  video name to VideoCapture object

  def _openVideo (self, videopath):
    ''' Open video and set up bookkeeping '''
    logging.debug ('opening video: %s' % videopath)
    videopath = op.join(self.rootdir, videopath)
    if not op.exists(videopath):
      raise ValueError('videopath does not exist: %s' % videopath)
    handle = imageio.get_reader(videopath)
    return handle

  def readImpl (self, image_id, ismask):
    # choose the dictionary, depending on whether it's image or mask
    video_dict = self.image_video if ismask else self.mask_video
    # video id set up
    videopath = op.dirname(image_id)
    if videopath not in video_dict:
      video_dict[videopath] = self._openVideo (videopath)
    # frame id
    frame_name = op.basename(image_id)
    try:
      frame_id = int(frame_name)  # number
    except ValueError:
      raise ValueError('Frame id is not a number in "%s"' % image_id)
    if frame_id < 0:
      raise ValueError('frame_id is %d, but can not be negative.' % frame_id)
    logging.debug ('from image_id %s, got frame_id %d' % (image_id, frame_id))
    # read the frame
    if frame_id >= video_dict[videopath].get_length():
      raise ValueError('frame_id %d exceeds the video length' % frame_id)
    img = video_dict[videopath].get_data(frame_id)
    img = np.asarray(img)
    # assign the dict back to where it was taken from
    if ismask: self.mask_video = video_dict 
    else: self.image_video = video_dict
    # and finally...
    return img

  def imread (self, image_id):
    if image_id in self.image_cache: 
        logging.debug ('imread: found image in cache')
        return self.image_cache[image_id]  # get cached image if possible
    image = self.readImpl (image_id, ismask=False)
    logging.debug ('imread: new image, updating cache')
    self.image_cache = {image_id: image}   # currently only 1 image in the cache
    return image

  def maskread (self, mask_id):
    if mask_id is None:
        return None
    if mask_id in self.mask_cache: 
        logging.debug ('maskread: found mask in cache')
        return self.mask_cache[mask_id]  # get cached mask if possible
    mask = self.readImpl (mask_id, ismask=True)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    logging.debug ('imread: new mask, updating cache')
    self.mask_cache = {mask_id: mask}   # currently only 1 image in the cache
    return mask

  def close(self):
    for key in self.image_video:
      self.image_video[key].close()
    for key in self.mask_video:
      self.mask_video[key].close()



class VideoWriter:

  def __init__(self, rootdir='.', vimagefile=None, vmaskfile=None, overwrite=False, fps=1):
    '''
    Args:
      rootdir:     imagefile and maskfile are considered paths relative to rootdir.
    '''
    self.overwrite  = overwrite
    self.vimagefile = vimagefile
    self.vmaskfile = vmaskfile
    self.image_writer = None
    self.mask_writer = None
    self.image_current_frame = -1
    self.mask_current_frame = -1
    self.frame_size = None        # used for checks
    self.fps = fps
    self.rootdir = rootdir
    logging.info('Rootdir set to "%s"' % rootdir)
    
  def _openVideo (self, ref_frame, ismask):
    ''' open a video for writing with parameters from the reference video (from reader) '''
    width  = ref_frame.shape[1]
    height = ref_frame.shape[0]
    if self.frame_size is None:
      self.frame_size = (width, height)
    else:
      assert self.frame_size == (width, height), \
         'frame_size different for image and mask: %s vs %s' % \
         (self.frame_size, (width, height))

    vpath = op.join(self.rootdir, self.vmaskfile) if ismask else op.join(self.rootdir, self.vimagefile)
    logging.info ('Opening video: %s' % vpath)
    
    # check if video exists
    if op.exists (vpath):
      if self.overwrite:
        os.remove(vpath)
      else:
        raise FileExistsError('Video already exists: %s. A mistake?' % vpath)
        
    # check if dir exists
    if not op.exists(op.dirname(vpath)):
      os.makedirs(op.dirname(vpath))

    if ismask:
      # "huffyuv" codec is good for png. "rgb24" keeps all colors and is supported by VLC.
      self.mask_writer = imageio.get_writer(vpath, fps=self.fps,
        codec='huffyuv', pixelformat='rgb24')
    else:
      # "mjpeg" for JPG images with highest quality and keeping all info with "yuvj444p".
      self.image_writer = imageio.get_writer(vpath, fps=self.fps,
        codec='mjpeg', quality=10, pixelformat='yuvj444p')

  def imwrite (self, image):
    # Multiple checks and lazy init.
    assert self.vimagefile is not None
    if self.image_writer is None:
      self._openVideo (image, ismask=False)
    assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
    assert (image.shape[1], image.shape[0]) == self.frame_size
    # Write.
    self.image_writer.append_data(image)
    # Return recorded imagefile.
    self.image_current_frame += 1
    return op.relpath('%s/%06d' % (op.splitext(self.vimagefile)[0], self.image_current_frame), self.rootdir)

  def maskwrite (self, mask):
    # Multiple checks and lazy init.
    assert self.vmaskfile is not None
    assert len(mask.shape) == 2
    if self.mask_writer is None:
      self._openVideo (mask, ismask=True)
    assert mask.dtype == np.uint8
    if len(mask.shape) == 2:
      mask = np.stack((mask, mask, mask), axis=-1)  # Otherwise mask is not written well.
    assert len(mask.shape) == 3 and mask.shape[2] == 3, mask.shape
    assert (mask.shape[1], mask.shape[0]) == self.frame_size
    # write.
    self.mask_writer.append_data(mask)
    # Return recorded maskfile.
    self.mask_current_frame += 1
    return op.relpath('%s/%06d' % (op.splitext(self.vmaskfile)[0], self.mask_current_frame), self.rootdir)

  def close (self):
    if self.image_writer is not None: 
      self.image_writer.close()
    if self.mask_writer is not None: 
      self.mask_writer.close()


class PictureReader:
  ''' Implementation of imagery reader based on "Image" <-> "Picture file (.jpg, .png, etc)". '''

  def __init__(self, rootdir):
    logging.debug('Creating PictureReader with rootdir: %s' % rootdir)
    self.rootdir = rootdir

  def _readImpl (self, image_id):
    imagepath = op.join(self.rootdir, image_id)
    logging.debug ('imagepath: %s, rootdir: %s' % (imagepath, self.rootdir))
    if not op.exists (imagepath):
      raise ValueError('Image does not exist at path: "%s"' % imagepath)
    try:
      return imageio.imread(imagepath)
    except ValueError:
      raise ValueError('PictureReader failed to read image_id %s at rootdir %s.'
        % (image_id, self.rootdir))

  def imread (self, image_id):
    return self._readImpl(image_id)
 
  def maskread (self, mask_id):
    return self._readImpl (mask_id)

  def close(self):
    pass


class PictureWriter:

  def __init__(self, rootdir='.', jpg_quality=100):
    self.jpg_quality = jpg_quality
    self.rootdir = rootdir

  def _writeImpl (self, image_id, image):
    if image is None:
      raise ValueError('image to write is None')
    imagepath = op.join(self.rootdir, image_id)
    if not op.exists (op.dirname(imagepath)):
      os.makedirs (op.dirname(imagepath))

    if op.splitext(imagepath)[1] in ['.jpg', '.jpeg']:
      imageio.imwrite (imagepath, image, quality=self.jpg_quality)
    else:
      imageio.imwrite (imagepath, image)

  def imwrite (self, imagepath, image):
    self._writeImpl (imagepath, image)

  def maskwrite (self, maskpath, mask):
    self._writeImpl (maskpath, mask)

  def close (self):
    pass


class ImageryReader:
  ''' A wrapper class around PictureReader and VideoReader.
  The purpose is to automatically figure out if an image_id is a picture or video frame.
  If it is a picture, create PictureReader. If it is a video frame, create VideoReader.
  '''

  def __init__(self, rootdir):  # TODO: pass kwargs to self.reader.__init__
    self.rootdir = rootdir
    if not isinstance(rootdir, str):
      raise ValueError('rootdir must be a string, got "%s"' % str(rootdir))
    self.reader = None  # Lazy ionitialization.

  def close(self):
    if self.reader is not None:
      self.reader.close()

  def imread(self, image_id):
    logging.debug('Try to read image_id "%s"' % image_id)
    if self.reader is not None:
      return self.reader.imread(image_id)

    try:
      self.reader = PictureReader(rootdir=self.rootdir)
      return self.reader.imread(image_id)
    except Exception:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      logging.debug(pformat(traceback.format_exception(exc_type, exc_value, exc_traceback)))
      logging.debug('Seems like it is not a picture.')

    try:
      self.reader = VideoReader(rootdir=self.rootdir)
      return self.reader.imread(image_id)
    except Exception:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      logging.debug(pformat(traceback.format_exception(exc_type, exc_value, exc_traceback)))
      logging.debug('Seems like it is not a video.')

    raise TypeError('The provided image_id "%s" with rootdir "%s" does not seem to refer to '
      'either picture file or video frame.' % (image_id, self.rootdir))

  def maskread(self, mask_id):
    if self.reader is not None:
      return self.reader.maskread(mask_id)

    try:
      self.reader = PictureReader(rootdir=self.rootdir)
      return self.reader.maskread(mask_id)
    except Exception as e:
      logging.debug('Seems like it is not a picture. Exception: "%s"' % e)

    try:
      self.reader = VideoReader(rootdir=self.rootdir)
      return self.reader.maskread(mask_id)
    except Exception as e:
      logging.debug('Seems like it is not a video. Exception: "%s"' % e)

    raise TypeError('mask_id "%s" and rootdir "%s" does not refer to picture or video frame.' % (mask_id, self.rootdir))
