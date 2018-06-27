import os, sys, os.path as op
import numpy as np
import cv2
import logging
import struct
from pkg_resources import parse_version


def getImageSize(file_path):
    """
    Return (height, width) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)
    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)
        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise Exception("UnknownImageFormat. StructError" + msg)
            except ValueError:
                raise Exception("UnknownImageFormat. ValueError" + msg)
            except Exception as e:
                raise Exception(e.__class__.__name__ + msg)
        else:
            raise Exception(
                "Sorry, don't know how to get information from this file."
            )
    return height, width



def validateMask(mask):
  if mask is None:
    raise ValueError('Mask is None.')
  if not isinstance(mask, np.ndarray):
    raise ValueError('Mask is of type %s' % type(mask))
  if len(mask.shape) != 2:
    raise ValueError('Mask is not grayscale, it has %d channels' % len(mask.shape))
  if mask.dtype != np.uint8:
    raise ValueError('Mask is not 8-bit, it is type %s' % mask.dtype)
  return mask


def _is_video(image_id):
  relpath = os.getenv('HOME')
  videopath = op.join(relpath, op.dirname(image_id))
  if not op.exists(videopath):
    return False
  try:
    int(op.basename(image_id))  # Check if the frame_id is a number.
    return True
  except ValueError:
    return False

def _is_image(image_id):
  relpath = os.getenv('HOME')
  imagepath = op.join(relpath, image_id)
  if op.exists(imagepath):
    return True
  else:
    return False

def imread(imagefile):
  global reader
  try:
    reader
  except NameError:
    # Create global 'reader' if it does not exist yet.
    if _is_video(imagefile):
      reader = VideoReader()
    elif _is_image(imagefile):
      reader = PictureReader()
    else:
      raise ValueError('imagefile refers neither to image not to video: %s' % imagefile)
  return reader.imread(imagefile)

def maskread(imagefile):
  global reader
  try:
    reader
  except NameError:
    # Create global 'reader' if it does not exist yet.
    if _is_video(imagefile):
      reader = VideoReader()
    elif _is_image(imagefile):
      reader = PictureReader()
    else:
      raise ValueError('imagefile refers neither to image not to video: %s' % imagefile)
  return reader.maskread(imagefile)



# returns OpenCV VideoCapture property id given, e.g., "FPS"
def capPropId(prop):
  OPCV3 = parse_version(cv2.__version__) >= parse_version('3')
  return getattr(cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

def getVideoLength(video_cv2):
  return int(video_cv2.get(capPropId('FRAME_COUNT')))


class VideoReader:
  '''Implementation based on Image <-> Frame in video.'''

  def __init__ (self):
    self.relpath = os.getenv('HOME')
    if self.relpath is not None:
      logging.info('ReaderVideo: relpath is set to %s' % self.relpath)
    else:
      logging.debug('ReaderVideo: relpath is NOT set.')
    self.image_cache = {}    # cache of previously read image(s)
    self.mask_cache = {}     # cache of previously read mask(s)
    self.image_video = {}    # map from image video name to VideoCapture object
    self.mask_video = {}     # map from mask  video name to VideoCapture object

  def _openVideoCapture_ (self, videopath):
    ''' Open video and set up bookkeeping '''
    logging.debug ('opening video: %s' % videopath)
    if self.relpath is not None:
      videopath = op.join(self.relpath, videopath)
      logging.debug('Videopath is relative to the provided relpath: %s' % videopath)
    if not op.exists(videopath):
      raise Exception('videopath does not exist: %s' % videopath)
    handle = cv2.VideoCapture(videopath)  # open video
    if not handle.isOpened():
        raise Exception('video failed to open: %s' % videopath)
    return handle

  def readImpl (self, image_id, ismask):
    # choose the dictionary, depending on whether it's image or mask
    video_dict = self.image_video if ismask else self.mask_video
    # video id set up
    videopath = op.dirname(image_id)
    if videopath not in video_dict:
      video_dict[videopath] = self._openVideoCapture_ (videopath)
    # frame id
    frame_name = op.basename(image_id)
    frame_id = int(filter(lambda x: x.isdigit(), frame_name))  # number
    logging.debug ('from image_id %s, got frame_id %d' % (image_id, frame_id))
    # read the frame
    video_dict[videopath].set(capPropId('POS_FRAMES'), frame_id)
    retval, img = video_dict[videopath].read()
    if not retval:
      if frame_id >= video_dict[videopath].get(capPropId('FRAME_COUNT')):
        raise ValueError('frame_id %d exceeds the video length' % frame_id)
      else:
        raise Exception('could not read image_id %s' % image_id)
    # assign the dict back to where it was taken from
    if ismask: self.mask_video = video_dict 
    else: self.image_video = video_dict
    # and finally...
    return img

  def imread (self, image_id):
    if image_id in self.image_cache: 
        logging.debug ('imread: found image in cache')
        return self.image_cache[image_id]  # get cached image if possible
    image = self.readImpl (image_id, ismask=False)[:,:,::-1]
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


class VideoWriter:

  def __init__(self, vimagefile=None, vmaskfile=None, overwrite=False, fps=2):
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
    fourcc = 1196444237
    width  = ref_frame.shape[1]
    height = ref_frame.shape[0]
    if self.frame_size is None:
      self.frame_size = (width, height)
    else:
      assert self.frame_size == (width, height), \
         'frame_size different for image and mask: %s vs %s' % \
         (self.frame_size, (width, height))

    vpath = self.vmaskfile if ismask else self.vimagefile
    logging.info ('SimpleWriter: opening video: %s' % vpath)
    
    # check if video exists
    if op.exists (vpath):
      if self.overwrite:
        os.remove(vpath)
      else:
        raise Exception('Video already exists: %s. A mistake?' % vpath)
        
    # check if dir exists
    if not op.exists(op.dirname(vpath)):
      if self.overwrite:
        os.makedirs(op.dirname(vpath))
      else:
        raise Exception('Video dir does not exist: %s. A mistake?' % op.dirname(vpath))

    handler = cv2.VideoWriter (vpath, fourcc, self.fps, self.frame_size, isColor=True)#not ismask)
    if not handler.isOpened():
        raise Exception('video failed to open: %s' % videopath)
    if ismask:
        self.mask_writer  = handler
    else:
        self.image_writer = handler

  def imwrite (self, image):
    assert self.vimagefile is not None
    if self.image_writer is None:
      self._openVideo (image, ismask=False)
    assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
    # write
    assert (image.shape[1], image.shape[0]) == self.frame_size
    self.image_writer.write(image[:,:,::-1])
    # return recorded imagefile
    self.image_current_frame += 1
    return '%s/%06d' % (op.splitext(self.vimagefile)[0], self.image_current_frame)

  def maskwrite (self, mask):
    assert self.vmaskfile is not None
    assert len(mask.shape) == 2
    assert mask.dtype == bool
    if self.mask_writer is None:
      self._openVideo (mask, ismask=True)
    mask = mask.copy().astype(np.uint8) * 255
    mask = np.stack((mask, mask, mask), axis=-1)  # Otherwise mask is not written well.
    # write
    assert len(mask.shape) == 3 and mask.shape[2] == 3, mask.shape
    assert (mask.shape[1], mask.shape[0]) == self.frame_size
    self.mask_writer.write(mask)
    # return recorded imagefile
    self.mask_current_frame += 1
    return '%s/%06d' % (op.splitext(self.vmaskfile)[0], self.mask_current_frame)

  def close (self):
    if self.image_writer is not None: 
      self.image_writer.release()
    if self.mask_writer is not None: 
      self.mask_writer.release()



class PictureReader:

  def __init__ (self):
    #self.image_cache = {}   # cache of previously read image(s)
    #self.mask_cache = {}    # cache of previously read mask(s)
    pass

  def _readImpl (self, image_id):
    imagepath = op.join (os.getenv('HOME'), image_id)
    logging.debug ('imagepath: %s' % imagepath)
    if not op.exists (imagepath):
      raise Exception ('image does not exist at path: "%s"' % imagepath)
    img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    if img is None:
      raise IOError ('image file exists, but failed to read it')
    return img

  def imread (self, image_id):
    #if image_id in self.image_cache: 
    #  logging.debug ('imread: found image in cache')
    #  return self.image_cache[image_id]  # get cached image if possible
    image = self._readImpl(image_id)[:,:,::-1]
    #logging.debug ('imread: new image, updating cache')
    #self.image_cache = {image_id: image}   # currently only 1 image in the cache
    return image

  def maskread (self, mask_id):
    #if mask_id in self.mask_cache: 
    #  logging.debug ('maskread: found mask in cache')
    #  return self.mask_cache[mask_id]  # get cached mask if possible
    mask = self._readImpl (mask_id)
    if len(mask.shape) == 3:
      logging.warning('PictureReader: mask file should be grayscale: %s' % mask_id)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #logging.debug ('imread: new mask, updating cache')
    #self.mask_cache = {mask_id: mask}   # currently only 1 image in the cache
    return mask

  def close(self):
    pass


class PictureWriter:

  def _writeImpl (self, image, image_id):
    imagepath = op.join (os.getenv('CITY_PATH'), image_id)
    if image is None:
      raise Exception ('image to write is None')
    if not op.exists (op.dirname(imagepath)):
      os.makedirs (op.dirname(imagepath))
    cv2.imwrite (imagepath, image)

  def imwrite (self, image, image_id):
    assert len(image.shape) == 3 and image.shape[2] == 3
    self._writeImpl(image[:,:,::-1], image_id)

  def maskwrite (self, mask, mask_id):
    assert len(mask.shape) == 2
    assert mask.dtype == bool
    mask = mask.copy().astype(np.uint8) * 255
    self._writeImpl (mask, mask_id)

  def close (self):
    pass
