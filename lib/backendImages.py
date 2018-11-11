import os, sys, os.path as op
import numpy as np
import imageio
import logging
import struct
import cv2
import traceback
from pprint import pformat
from PIL import Image
from pkg_resources import parse_version


def getPictureSize(imagepath):
  if not op.exists (imagepath):
    raise ValueError('Image does not exist at path: "%s"' % imagepath)
  im = Image.open(imagepath)
  width, height = im.size
  return height, width


# returns OpenCV VideoCapture property id given, e.g., "FPS"
def capOpencvPropId(prop):
  OPCV3 = parse_version(cv2.__version__) >= parse_version('3')
  return getattr(cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

def _getOpencvVideoLength(video_cv2):
  return int(video_cv2.get(capOpencvPropId('FRAME_COUNT')))


class VideoReader:
  '''Implementation of imagery reader based on "Image" <-> "Frame in video".'''

  def __init__ (self, rootdir, middleware='imageio'):
    '''
    Args:
      rootdir:     imagefile and maskfile are considered paths relative to rootdir.
      middleware:  a layer between this class and ffmpeg.
                   "opencv":   pros: can have any number of open videos
                               cons: does not support huffyuv codec, dumps ffmpeg warning logs.
                   "imageio":  pros: supports huffyuv codec, can control the verbosity of ffmpeg logs.
                               cons: allows only a limited number of open videos.
    '''
    self.rootdir = rootdir
    logging.info('Root is set to %s' % self.rootdir)
    self.image_cache = {}    # cache of previously read image(s)
    self.mask_cache = {}     # cache of previously read mask(s)
    self.image_video = {}    # map from image video name to VideoCapture object
    self.mask_video = {}     # map from mask  video name to VideoCapture object
    if middleware not in ['opencv', 'imageio']:
      raise ValueError('Only "opencv" and "imageio" for middleware are supported.')
    self.middleware = middleware

  def _openVideo (self, videopath):
    ''' Open video and set up bookkeeping '''
    logging.debug ('opening video: %s' % videopath)
    videopath = op.join(self.rootdir, videopath)
    if not op.exists(videopath):
      raise ValueError('videopath does not exist: %s' % videopath)
    if self.middleware == 'imageio':
      handle = imageio.get_reader(videopath)
    elif self.middleware == 'opencv':
      handle = cv2.VideoCapture(videopath)  # open video
      if not handle.isOpened():
          raise ValueError('video failed to open: %s' % videopath)
    else:
      assert False
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
    if self.middleware == 'imageio':
      if frame_id >= video_dict[videopath].get_length():
        raise ValueError('frame_id %d exceeds the video length' % frame_id)
      img = video_dict[videopath].get_data(frame_id)
      img = np.asarray(img)
    elif self.middleware == 'opencv':
      video_dict[videopath].set(capOpencvPropId('POS_FRAMES'), frame_id)
      retval, img = video_dict[videopath].read()
      if not retval:
        # Trying this after reading to save time while not storing video lengths.
        if frame_id >= video_dict[videopath].get(capOpencvPropId('FRAME_COUNT')):
          raise ValueError('frame_id %d exceeds the video length' % frame_id)
        else:
          raise Exception('could not read image_id %s' % image_id)
      img = img[:,:,::-1].copy()  # Copy needed to fix the order.

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

  def __init__(self, rootdir='.', vimagefile=None, vmaskfile=None, overwrite=False, fps=1,
               middleware='imageio', imagecodec=None, maskcodec=None):
    '''
    Args:
      rootdir:     imagefile and maskfile are considered paths relative to rootdir.
      middleware:  a layer between this class and ffmpeg.
                   "opencv":   pros: can have any number of open videos
                               cons: does not support huffyuv codec, dumps ffmpeg warning logs.
                   "imageio":  pros: supports huffyuv codec, can control the verbosity of ffmpeg logs.
                               cons: allows only a limited number of open videos.
      imagecodec:  codec for image video.
      maskcodec:   codec for mask video. "Huffyuv" is a good choice for PNG-like masks.
         For "opencv" middleware, a codec must be a 4-letter string, e.g. "MJPG".
         For "imageio" middleware, a codec must be as it seen by ffmpeg, e.g. "mjpeg" or "huffyuv"
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
    self.middleware = middleware
    if self.middleware == 'imageio':
      self.imagecodec = imagecodec if imagecodec is not None else 'mjpeg'
      self.maskcodec = maskcodec if maskcodec is not None else 'huffyuv'
    elif self.middleware == 'opencv':
      def getFourcc(name):
        if len(name) is not 4:
          raise ValueError('For opencv middleware, the codec name must be of 4 letters.')
        return cv2.VideoWriter_fourcc(*name.upper())
      self.imagecodec = getFourcc(imagecodec if imagecodec is not None else 'mjpg')
      self.maskcodec = getFourcc(maskcodec if maskcodec is not None else 'mjpg')
    else:
      raise ValueError('Only "opencv" and "imageio" for middleware are supported.')
    

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

    if self.middleware == 'imageio':
      if ismask:
        self.mask_writer = imageio.get_writer(vpath, codec=self.maskcodec, fps=self.fps, quality=10, pixelformat='gray')
      else:
        self.image_writer = imageio.get_writer(vpath, codec=self.imagecodec, fps=self.fps, quality=10, pixelformat='yuvj444p')
    elif self.middleware == 'opencv':
      if ismask:
        self.mask_writer = cv2.VideoWriter (vpath, self.maskcodec, self.fps, self.frame_size, isColor=True)
      else:
        self.image_writer = cv2.VideoWriter (vpath, self.imagecodec, self.fps, self.frame_size, isColor=True)
    else:
      assert False

  def imwrite (self, image):
    # Multiple checks and lazy init.
    assert self.vimagefile is not None
    if self.image_writer is None:
      self._openVideo (image, ismask=False)
    assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
    assert (image.shape[1], image.shape[0]) == self.frame_size
    # Write.
    if self.middleware == 'opencv':
      self.image_writer.write(image[:,:,::-1])
    elif self.middleware == 'imageio':
      self.image_writer.append_data(image)
    else:
      assert False
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
    if self.middleware == 'opencv':
      self.mask_writer.write(mask)
    elif self.middleware == 'imageio':
      self.mask_writer.append_data(mask)
    else:
      assert False
    # Return recorded maskfile.
    self.mask_current_frame += 1
    return op.relpath('%s/%06d' % (op.splitext(self.vmaskfile)[0], self.mask_current_frame), self.rootdir)

  def close (self):
    if self.middleware == 'imageio':
      if self.image_writer is not None: 
        self.image_writer.close()
      if self.mask_writer is not None: 
        self.mask_writer.close()
    elif self.middleware == 'opencv':
      if self.image_writer is not None: 
        self.image_writer.release()
      if self.mask_writer is not None: 
        self.mask_writer.release()


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

  def __init__(self, rootdir, middleware=None):  # TODO: pass kwargs to self.reader.__init__
    self.rootdir = rootdir
    if not isinstance(rootdir, str):
      raise ValueError('rootdir must be a string, got "%s"' % str(rootdir))
    self.middleware = middleware
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
      self.reader = VideoReader(rootdir=self.rootdir, middleware=self.middleware)
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
