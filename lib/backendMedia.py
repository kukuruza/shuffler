import os, sys, os.path as op
import numpy as np
import imageio
import logging
import shutil
import cv2
import traceback
from pprint import pformat
from operator import itemgetter
from PIL import Image


'''
The support for reading and writing media in the form of single files and
multiple files, which are supported by our backend imageio:
https://imageio.readthedocs.io/en/stable/formats.html

Levels of abstraction:

- classes VideoReader and PictureReader:
  1) hides implementation if "imread" for images and "maskread" for masks,
  2) maintain cache of images already in memory,
  3) maintain a collection of open videos (in case of VideoReader).

- class MediaReader:
  1) hides whether VideoReader or PictureReader should be used,
  2) compute absolute paths from "rootdir" and the provided paths.


- classes VideoWriter and PictureWriter:
  1) create / recreate a media / dir or raise an exception,
  2) hides implementation if "imwrite" for images and "maskwrite" for masks,
  3) counts frames / images (or writes the given name in case of PictureWriter.)

- class MediaWriter:
  1) hides whether VideoWriter or PictureWriter should be used,
  2) compute recorded paths relative to "rootdir".

'''

def normalizeSeparators(path):
  ''' Replace mixed slashes to forward slashes in a path.
  Provides the compatibility for Windows. '''
  return path.replace('\\', '/')


def getPictureSize(imagepath):
  if not op.exists (imagepath):
    raise ValueError('Image does not exist at path: "%s"' % imagepath)
  im = Image.open(imagepath)
  width, height = im.size
  return height, width


class VideoReader:
  '''Implementation of imagery reader based on "Image" <-> "Frame in video".'''

  def __init__ (self):
    self.image_cache = {}    # cache of previously read image(s)
    self.mask_cache = {}     # cache of previously read mask(s)
    self.videos = {}         # map from video name to imageio video object
    self.used_ago = {}       # map from video name to number of frames since the last time it was used

  def _openVideo (self, videopath):
    ''' Open video and set up bookkeeping '''
    videopath = normalizeSeparators(videopath)
    logging.debug ('opening video: %s' % videopath)
    if not op.exists(videopath):
      raise ValueError('videopath does not exist: %s' % videopath)
    handle = imageio.get_reader(videopath)
    return handle

  def readImpl (self, image_id, ismask):
    # video id set up
    videopath = op.dirname(image_id)
    if videopath not in self.videos:
      if len(self.videos) >= 10:
        # Find the file that was used the longest ago.
        keymax = max(self.used_ago.items(), key=itemgetter(1))[0]
        logging.debug('Open %d videos, closing video "%s" to release a handle.' %
          (len(self.used_ago), keymax))
        self.videos[keymax].close()
        del self.videos[keymax]
        del self.used_ago[keymax]
      # Repeat opening.
      self.videos[videopath] = self._openVideo (videopath)
      logging.debug('Have %d videos opened' % len(self.videos))
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
    if frame_id >= self.videos[videopath].get_length():
      raise ValueError('frame_id %d exceeds the video length' % frame_id)
    img = self.videos[videopath].get_data(frame_id)
    img = np.asarray(img)
    # increase everyones's "long ago", except the current video
    for key in self.used_ago:
      self.used_ago[key] += 1
    self.used_ago[videopath] = 0
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
    for key in self.videos:
      self.videos[key].close()



class VideoWriter:

  def __init__(self, vimagefile=None, vmaskfile=None, overwrite=False, fps=1):
    self.overwrite  = overwrite
    self.vimagefile = vimagefile
    self.vmaskfile = vmaskfile
    self.image_writer = None
    self.mask_writer = None
    self.image_current_frame = -1
    self.mask_current_frame = -1
    self.frame_size = None        # used for checks
    self.fps = fps
    
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

    vpath = self.vmaskfile if ismask else self.vimagefile
    vpath = normalizeSeparators(vpath)
    logging.info ('Opening video: %s' % vpath)

    # Check if a user passed without extension (otherwise, the error is obscure.)
    if op.splitext(vpath)[1] == '':
      raise TypeError('Video path "%s" should have an extension.' % vpath)
    
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
    return '%s/%06d' % (self.vimagefile, self.image_current_frame)

  def maskwrite (self, mask):
    # Multiple checks and lazy init.
    assert self.vmaskfile is not None
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
    return '%s/%06d' % (self.vmaskfile, self.mask_current_frame)

  def close (self):
    if self.image_writer is not None: 
      self.image_writer.close()
    if self.mask_writer is not None: 
      self.mask_writer.close()


class PictureReader:
  ''' Implementation of imagery reader based on "Image" <-> "Picture file (.jpg, .png, etc)". '''

  def _readImpl (self, image_id):
    image_id = normalizeSeparators(image_id)
    logging.debug ('image_id: %s' % image_id)
    if not op.exists (image_id):
      raise ValueError('Image does not exist at path: "%s"' % image_id)
    try:
      return imageio.imread(image_id)
    except ValueError:
      raise ValueError('PictureReader failed to read image_id %s.' % image_id)

  def imread (self, image_id):
    return self._readImpl(image_id)
 
  def maskread (self, mask_id):
    return self._readImpl (mask_id)

  def close(self):
    pass


class PictureWriter:

  def __init__(self, imagedir=None, maskdir=None, overwrite=False, jpg_quality=100):
    self.jpg_quality = jpg_quality
    self.overwrite = overwrite
    self.imagedir = imagedir
    self.maskdir = maskdir
    self.image_current_frame = -1
    self.mask_current_frame = -1

    if self.imagedir is not None:
      # Raise, if imagedir is specified and exists and overwrite is disabled.
      if op.exists(imagedir) and not self.overwrite:
        raise ValueError('Directory already exists: %s' % self.imagedir)
      # Create imagedir, if it is specified and does not exist.
      elif not op.exists(self.imagedir):
        logging.debug('PictureWriter will create imagedir "%s"' % self.imagedir)
        os.makedirs (imagedir)
      # Delete and recreate imagedir, if it is specified and exists.
      elif op.exists(self.imagedir) and self.overwrite:
        logging.debug('PictureWriter will recreate imagedir "%s"' % self.imagedir)
        shutil.rmtree (imagedir)
        os.makedirs (imagedir)

    if self.maskdir is not None:
      # Raise, if imagedir is specified and exists and overwrite is disabled.
      if op.exists(maskdir) and not self.overwrite:
        raise ValueError('Directory already exists: %s' % self.maskdir)
      # Create imagedir, if it is specified and does not exist.
      elif not op.exists(self.maskdir):
        logging.debug('PictureWriter will create maskdir "%s"' % self.maskdir)
        os.makedirs (maskdir)
      # Delete and recreate maskdir, if it is specified and exists.
      elif op.exists(self.maskdir) and self.overwrite:
        logging.debug('PictureWriter will recreate maskdir "%s"' % self.maskdir)
        shutil.rmtree (maskdir)
        os.makedirs (maskdir)

  def imwrite (self, image, namehint=None):
    if self.imagedir is None:
      raise ValueException('Tried to write an image, but imagedir was not specified at init.')
    if namehint is not None and not isinstance(namehint, str):
      raise ValueError('namehint must be a string, got %s' % str(namehint))

    # If "namehint" is not specified, compute name as the next frame.
    if namehint is None:
      self.image_current_frame += 1
      name = '%06d.jpg' % self.image_current_frame
    else:
      name = op.basename(namehint)
      # Add extension if not specified in "name"
      if not op.splitext(name)[1]:
        name = '%s.jpg' % name

    # Compute path from name.
    imagepath = op.join(self.imagedir, name)
    imagepath = normalizeSeparators(imagepath)
    logging.debug ('Writing image to path: "%s"' % imagepath)
    if op.exists(imagepath) and not self.overwrite:
      raise Exception('Imagepath has been already recorded before: "%s"')

    # Write.
    imageio.imwrite (imagepath, image, quality=self.jpg_quality)
    return imagepath

  def maskwrite (self, mask, namehint=None):
    if self.maskdir is None:
      raise ValueException('Tried to write an mask, but maskdir was not specified at init.')

    # If "namehint" is not specified, compute name as the next frame.
    if namehint is None:
      self.mask_current_frame += 1
      name = '%06d.png' % self.mask_current_frame
    else:
      name = op.basename(namehint)
      # Add extension if not specified in "name"
      if not op.splitext(name)[1]:
        name = '%s.png' % name

    # Compute path from name.
    maskpath = op.join(self.maskdir, name)
    maskpath = normalizeSeparators(maskpath)
    logging.debug ('Writing mask to path: "%s"' % maskpath)
    if op.exists(maskpath) and not self.overwrite:
      raise Exception('Maskpath has been already recorded before: "%s"')

    # Write.
    imageio.imwrite (maskpath, mask)
    return maskpath

  def close (self):
    pass


class MediaReader:
  ''' A wrapper class around PictureReader and VideoReader.
  The purpose is to automatically figure out if an image_id is a picture or video frame.
  If it is a picture, create PictureReader. If it is a video frame, create VideoReader.
  '''

  def __init__(self, rootdir):  # TODO: pass kwargs to self.reader.__init__
    self.rootdir = rootdir
    if not isinstance(rootdir, str):
      raise ValueError('rootdir must be a string, got %s' % str(rootdir))
    self.reader = None  # Lazy ionitialization.

  def close(self):
    if self.reader is not None:
      self.reader.close()

  def imread(self, image_id):
    logging.debug('Try to read image_id "%s" with rootdir "%s"' % (image_id, self.rootdir))
    image_id = op.join(self.rootdir, image_id)

    if self.reader is not None:
      return self.reader.imread(image_id)

    try:
      self.reader = PictureReader()
      return self.reader.imread(image_id)
    except Exception:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      logging.debug(pformat(traceback.format_exception(exc_type, exc_value, exc_traceback)))
      logging.debug('Seems like it is not a picture.')

    try:
      self.reader = VideoReader()
      return self.reader.imread(image_id)
    except Exception:
      exc_type, exc_value, exc_traceback = sys.exc_info()
      logging.debug(pformat(traceback.format_exception(exc_type, exc_value, exc_traceback)))
      logging.debug('Seems like it is not a video.')

    raise TypeError('The provided image_id "%s" (rootdir "%s" was added) '
      'does not seem to refer to either picture file or video frame.' %
      (image_id, self.rootdir))

  def maskread(self, mask_id):
    logging.debug('Try to read mask_id "%s" with rootdir "%s"' % (mask_id, self.rootdir))
    mask_id = op.join(self.rootdir, mask_id)

    if self.reader is not None:
      return self.reader.maskread(mask_id)

    try:
      self.reader = PictureReader()
      return self.reader.maskread(mask_id)
    except Exception as e:
      logging.debug('Seems like it is not a picture. Exception: "%s"' % e)

    try:
      self.reader = VideoReader()
      return self.reader.maskread(mask_id)
    except Exception as e:
      logging.debug('Seems like it is not a video. Exception: "%s"' % e)

    raise TypeError('The provided mask_id "%s" (rootdir "%s" was added) '
      'does not seem to refer to either picture file or video frame.' %
      (mask_id, self.rootdir))


class MediaWriter:
  ''' A wrapper class around PictureWriter and VideoWriter. The purpose is
  1) to avoid if-else in the subcommand code, depending on how a user would like to record data.
  2) return paths relative to rootdir, if needed.
  '''

  def __init__(self, media_type, image_media=None, mask_media=None,
    rootdir='', overwrite=False):
    '''
    Args:
      media_type:   "video" for multiple images in ffmpeg formats,
                    "pictures" for directory with single images in ffmpeg formats.
      image_media:  path for "imagefile" to video or directory, depeding on media_type.
      mask_meda:    path for "maskfile" to video or directory, depeding on media_type.
      rootdir:      Output "imagefile" and "maskfile" will be computed relative to it.
                    Input image_media and mask_media are not affected.
      overwrite:    if media exists, overwrite it or raise an exception.
    '''
    if not isinstance(rootdir, str):
      raise ValueError('rootdir must be a string, got %s' % str(rootdir))

    self.media_type = media_type
    self.rootdir = rootdir
    self.picture_id = 0  # Only for media_type='pictures'.

    if media_type == 'video':
      self.writer = VideoWriter(vimagefile=image_media, vmaskfile=mask_media, overwrite=overwrite)
    elif media_type == 'pictures':
      self.writer = PictureWriter(imagedir=image_media, maskdir=mask_media, overwrite=overwrite)
    else:
      raise ValueError('"media" must be either "video" or "pictures", not %s' % media_type)

  def imwrite(self, image, namehint=None):
    ''' Args:
      namehint:  A file in directory will have basename(namehint), not a sequential number.
                 (Only for meda_type="pictures".)
    '''
    if self.media_type == 'video':
      image_id = self.writer.imwrite(image)
    elif self.media_type == 'pictures':
      image_id = self.writer.imwrite(image, namehint=namehint)

    return op.relpath(image_id, self.rootdir)

  def maskwrite(self, mask, namehint=None):
    ''' Args:
      namehint:  A file in directory will have basename(namehint), not a sequential number.
                 (Only for meda_type="pictures".)
    '''
    if self.media_type == 'video':
      mask_id = self.writer.maskwrite(mask)
    elif self.media_type == 'pictures':
      mask_id = self.writer.maskwrite(mask, namehint=namehint)

    return op.relpath(mask_id, self.rootdir)

  def close(self):
    self.writer.close()
