import os, os.path as op
import numpy as np
import imageio
import logging
import shutil
from operator import itemgetter
from PIL import Image as PILImage  # import PIL.Image does not work.

# The support for reading and writing media in the form of single files and
# multiple files, which are supported by our backend imageio:
# https://imageio.readthedocs.io/en/stable/formats.html

# Levels of abstraction:

# - classes VideoReader and PictureReader:
#   1) hides implementation if "imread" for images and "maskread" for masks,
#   2) maintain a collection of open videos (in case of VideoReader).

# - class MediaReader:
#   1) hides whether VideoReader or PictureReader should be used,
#   2) compute absolute paths from "rootdir" and the provided paths.
#   3) maintain cache of images already in memory,

# - classes VideoWriter and PictureWriter:
#   1) create / recreate a media / dir or raise an exception,
#   2) hides implementation if "imwrite" for images and "maskwrite" for masks,
#   3) counts frames / images (or writes the given name in case of PictureWriter.)

# - class MediaWriter:
#   1) hides whether VideoWriter or PictureWriter should be used,
#   2) compute recorded paths relative to "rootdir".

IMAGE_EXTENSIONS = [
    'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'
]
VIDEO_EXTENSIONS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def normalizeSeparators(path):
    ''' Replace mixed slashes to forward slashes in a path.
  Provides the compatibility for Windows. '''
    return path.replace('/', os.sep).replace('\\', os.sep)


def _getMediaType(image_id):
    '''
    Args:
      image_id: May correspond to either 'imagefile' or 'maskfile'.
                If the media is pictures, image_id = "path/to/image.jpg"
                If the media is a video, image_id = "path/to/video.avi/34"
    Returns:
      'PICTURE' or 'VIDEO'
    Raises:
      ValueError if can't figure out the type from image_id.
    '''

    def _MaybeRemoveLeadingDot(x):
        return x[1:] if x.startswith('.') else x

    mediadir = os.path.dirname(image_id)
    image_id_ext = _MaybeRemoveLeadingDot(os.path.splitext(image_id)[1])
    if image_id_ext.lower() in IMAGE_EXTENSIONS:
        return 'PICTURE'

    mediadir_ext = _MaybeRemoveLeadingDot(os.path.splitext(mediadir)[1])
    if image_id_ext == '' and mediadir_ext in VIDEO_EXTENSIONS:
        return 'VIDEO'
    raise ValueError(
        f'image_id "{image_id}" is neither a picture or a video frame.')


def getPictureHeightAndWidth(imagepath):
    if not op.exists(imagepath):
        raise ValueError(f'Image does not exist at path: "{imagepath}"')
    logging.debug('Get size of image "%s".', imagepath)
    im = PILImage.open(imagepath)
    width, height = im.size
    im.close()
    return height, width


class VideoReader:
    '''Implementation of imagery reader based on "Image" <-> "Frame in video".'''

    def __init__(self):
        # Map from video name to imageio video object.
        self.videos = {}
        # Map from video name to number of frames since the last time it was used.
        self.used_ago = {}
        # Video properties by video path.
        self.video_properties = {}

    def _openVideo(self, videopath):
        ''' Open video and set up bookkeeping '''
        videopath = normalizeSeparators(videopath)
        logging.debug('opening video: %s', videopath)
        if not op.exists(videopath):
            raise FileNotFoundError('videopath does not exist: %s' % videopath)
        handle = imageio.v2.get_reader(videopath)
        return handle

    def _getVideoHandle(self, image_id):
        # video id set up
        videopath = op.dirname(image_id)
        if videopath not in self.videos:
            if len(self.videos) >= 10:
                # Find the file that was used the longest ago.
                keymax = max(self.used_ago.items(), key=itemgetter(1))[0]
                logging.debug(
                    'Open %d videos, closing video "%s" to release a handle.',
                    len(self.used_ago), keymax)
                self.videos[keymax].close()
                del self.videos[keymax]
                del self.used_ago[keymax]
            self.videos[videopath] = self._openVideo(videopath)
            logging.debug('Have %d videos opened', len(self.videos))
        return self.videos[videopath], videopath

    def _getFrameId(self, image_id):
        video, videopath = self._getVideoHandle(image_id)
        # frame id
        frame_name = op.basename(image_id)
        try:
            frame_id = int(frame_name)  # number
        except ValueError as e:
            raise ValueError('Frame id is not a number in "%s"' %
                             image_id) from e
        if frame_id < 0:
            raise IndexError('frame_id is %d, but can not be negative.' %
                             frame_id)
        logging.debug('from image_id %s, got frame_id %d', image_id, frame_id)
        if frame_id >= video.count_frames():
            raise IndexError('frame_id %d exceeds the video length' % frame_id)

        # increase everyones's "long ago", except the current video
        for key in self.used_ago:
            self.used_ago[key] += 1
        self.used_ago[videopath] = 0
        # and finally...
        return videopath, frame_id

    def _readImpl(self, image_of_mask_id):
        videopath, frame_id = self._getFrameId(image_of_mask_id)
        img = self.videos[videopath].get_data(frame_id)
        img = np.asarray(img)
        return img

    def checkIdExists(self, image_of_mask_id):
        try:
            self._getFrameId(image_of_mask_id)
        except Exception as e:
            logging.info(
                'Image or mask id %s is not valid or does not exist. '
                'Got an exception:\n%s', image_of_mask_id, str(e))
            return False
        return True

    def getHeightAndWidth(self, image_or_mask_id):
        videopath = op.dirname(image_or_mask_id)
        if videopath not in self.video_properties:
            props = imageio.v3.improps(videopath)
        self.video_properties[videopath] = props
        print(props)
        return props.shape[1], props.shape[2]

    def imread(self, image_id):
        return self._readImpl(image_id)

    def maskread(self, mask_id):
        mask = self._readImpl(mask_id)
        # Take the 1st channel to make the mask grayscale.
        return mask[:, :, 0]

    def close(self):
        for key in self.videos:
            self.videos[key].close()


class VideoWriter:

    def __init__(self,
                 vimagefile=None,
                 vmaskfile=None,
                 overwrite=False,
                 fps=1):
        self.overwrite = overwrite
        self.vimagefile = vimagefile
        self.vmaskfile = vmaskfile
        self.image_writer = None
        self.mask_writer = None
        self.image_current_frame = -1
        self.mask_current_frame = -1
        self.frame_size = None  # used for checks
        self.fps = fps
        if vimagefile is not None and op.exists(vimagefile) and not overwrite:
            raise ValueError(
                'VideoWriter has to write to file "%s", '
                'but that file already exists. Pass overwrite=True.' %
                vimagefile)
        if vmaskfile is not None and op.exists(vmaskfile) and not overwrite:
            raise ValueError(
                'VideoWriter has to write to file "%s", '
                'but that file already exists. Pass overwrite=True.' %
                vmaskfile)

    def _openVideo(self, ref_frame, ismask):
        '''
        Open a video for writing with parameters from the reference video (from reader)
        '''
        width = ref_frame.shape[1]
        height = ref_frame.shape[0]
        if self.frame_size is None:
            self.frame_size = (width, height)
        else:
            assert self.frame_size == (width, height), \
               'frame_size different for image and mask: %s vs %s' % \
               (self.frame_size, (width, height))

        vpath = self.vmaskfile if ismask else self.vimagefile
        vpath = normalizeSeparators(vpath)
        logging.info('Opening video: %s', vpath)

        # Check if a user passed without extension (otherwise, the error is obscure.)
        if op.splitext(vpath)[1] == '':
            raise TypeError('Video path "%s" should have an extension.' %
                            vpath)

        # check if video exists
        if op.exists(vpath):
            if self.overwrite:
                os.remove(vpath)
            else:
                raise ValueError('Video already exists: %s. A mistake?' %
                                 vpath)

        # check if dir exists
        logging.debug('vpath: "%s", dirname: "%s"', vpath, op.dirname(vpath))
        if not op.exists(op.abspath(op.dirname(vpath))):
            os.makedirs(op.dirname(vpath))

        if ismask:
            # "huffyuv" codec is good for png. "rgb24" keeps all colors and is supported by VLC.
            self.mask_writer = imageio.get_writer(
                vpath,
                fps=self.fps,
                codec='huffyuv',
                macro_block_size=None,  # No resizing of frame.
                pixelformat='rgb24')
        else:
            # "mjpeg" for JPG images with highest quality and keeping all info with "yuvj444p".
            self.image_writer = imageio.get_writer(
                vpath,
                fps=self.fps,
                codec='mjpeg',
                quality=10,
                macro_block_size=None,  # No resizing of frame.
                pixelformat='yuvj444p')

    def imwrite(self, image):
        # Multiple checks and lazy init.
        assert self.vimagefile is not None
        # Open video in imwrite (lazily) in order to fetch image size.
        if self.image_writer is None:
            self._openVideo(image, ismask=False)
        assert len(image.shape) == 3 and image.shape[2] == 3, image.shape
        assert (image.shape[1], image.shape[0]) == self.frame_size, (
            'Unexpected image size: expected %s based on the first video frame, got %s.'
            % (str(self.frame_size), str(image.shape[0:2])))
        # Write.
        self.image_writer.append_data(image)
        # Return recorded imagefile.
        self.image_current_frame += 1
        return '%s/%d' % (self.vimagefile, self.image_current_frame)

    def maskwrite(self, mask):
        # Multiple checks and lazy init.
        assert self.vmaskfile is not None
        if self.mask_writer is None:
            self._openVideo(mask, ismask=True)
        assert mask.dtype == np.uint8
        if len(mask.shape) == 2:
            mask = np.stack((mask, mask, mask),
                            axis=-1)  # Otherwise mask is not written well.
        assert len(mask.shape) == 3 and mask.shape[2] == 3, mask.shape
        assert (mask.shape[1], mask.shape[0]) == self.frame_size
        # write.
        self.mask_writer.append_data(mask)
        # Return recorded maskfile.
        self.mask_current_frame += 1
        return '%s/%d' % (self.vmaskfile, self.mask_current_frame)

    def close(self):
        if self.image_writer is not None:
            self.image_writer.close()
        if self.mask_writer is not None:
            self.mask_writer.close()


class PictureReader:
    '''
    Implementation of imagery reader based on the one-to-one correspondence
    "Image" <-> "Picture file (.jpg, .png, etc)".
    '''

    def _readImpl(self, image_id):
        image_id = normalizeSeparators(image_id)
        logging.debug('image_id: %s', image_id)
        if not op.exists(image_id):
            logging.error('Real image path: %s', op.realpath(image_id))
            raise FileNotFoundError('Image does not exist at path: "%s"' %
                                    image_id)
        try:
            image = imageio.v2.imread(image_id)
            return np.asarray(image)
        except ValueError as e:
            raise ValueError('PictureReader failed to read image_id %s.' %
                             image_id) from e

    def checkIdExists(self, image_or_mask_id):
        return op.exists(image_or_mask_id)

    def getHeightAndWidth(self, image_or_mask_id):
        return getPictureHeightAndWidth(image_or_mask_id)

    def imread(self, image_id):
        return self._readImpl(image_id)

    def maskread(self, mask_id):
        return self._readImpl(mask_id)

    def close(self):
        pass


class PictureWriter:

    def __init__(self,
                 imagedir=None,
                 maskdir=None,
                 overwrite=False,
                 jpg_quality=100):
        self.jpg_quality = jpg_quality
        self.overwrite = overwrite
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.image_current_frame = -1
        self.mask_current_frame = -1

        if self.imagedir is not None:
            # Raise, if imagedir is specified and exists and overwrite is disabled.
            if op.exists(imagedir) and not self.overwrite:
                raise ValueError('Directory already exists: %s' %
                                 self.imagedir)
            # Create imagedir, if it is specified and does not exist.
            elif not op.exists(self.imagedir):
                logging.debug('PictureWriter will create imagedir "%s"',
                              self.imagedir)
                os.makedirs(imagedir)
            # Delete and recreate imagedir, if it is specified and exists.
            elif op.exists(self.imagedir) and self.overwrite:
                logging.debug('PictureWriter will recreate imagedir "%s"',
                              self.imagedir)
                shutil.rmtree(imagedir)
                os.makedirs(imagedir)

        if self.maskdir is not None:
            # Raise, if imagedir is specified and exists and overwrite is disabled.
            if op.exists(maskdir) and not self.overwrite:
                raise ValueError('Directory already exists: %s' % self.maskdir)
            # Create imagedir, if it is specified and does not exist.
            elif not op.exists(self.maskdir):
                logging.debug('PictureWriter will create maskdir "%s"',
                              self.maskdir)
                os.makedirs(maskdir)
            # Delete and recreate maskdir, if it is specified and exists.
            elif op.exists(self.maskdir) and self.overwrite:
                logging.debug('PictureWriter will recreate maskdir "%s"',
                              self.maskdir)
                shutil.rmtree(maskdir)
                os.makedirs(maskdir)

    def imwrite(self, image, namehint=None):
        '''
        Note: namehint may be a name e.g. "car.jpg" or path e.g. "cars/01.jpg".
        '''
        if self.imagedir is None:
            raise ValueError(
                'Tried to write an image, but imagedir was not specified at init.'
            )
        if namehint is not None and not isinstance(namehint, str):
            raise ValueError('namehint must be a string, got %s' %
                             str(namehint))

        # If "namehint" is not specified, compute name as the next frame.
        if namehint is None:
            self.image_current_frame += 1
            name = '%d.jpg' % self.image_current_frame
        else:
            name = namehint
            # Add extension if not specified in "name"
            if not op.splitext(name)[1]:
                name = '%s.jpg' % name

        # Compute path from name.
        imagepath = op.join(self.imagedir, name)
        imagepath = normalizeSeparators(imagepath)
        logging.debug('Writing image to path: "%s"', imagepath)
        if op.exists(imagepath) and not self.overwrite:
            raise Exception('Imagepath has been already recorded before: "%s"')

        # Process the case when namehint was a path.
        if namehint is not None and '/' in namehint:
            imagedir = op.dirname(imagepath)
            if not op.exists(imagedir):
                os.makedirs(imagedir)

        # Write.
        if op.splitext(name)[1] == '.jpg':
            imageio.imwrite(imagepath, image, quality=self.jpg_quality)
        else:
            imageio.imwrite(imagepath, image)
        return imagepath

    def maskwrite(self, mask, namehint=None):
        if self.maskdir is None:
            raise ValueError(
                'Tried to write an mask, but maskdir was not specified at init.'
            )

        # If "namehint" is not specified, compute name as the next frame.
        if namehint is None:
            self.mask_current_frame += 1
            name = '%d.png' % self.mask_current_frame
        else:
            name = namehint
            # Add extension if not specified in "name"
            if not op.splitext(name)[1]:
                name = '%s.png' % name

        # Compute path from name.
        maskpath = op.join(self.maskdir, name)
        maskpath = normalizeSeparators(maskpath)
        logging.debug('Writing mask to path: "%s"', maskpath)
        if op.exists(maskpath) and not self.overwrite:
            raise Exception('Maskpath has been already recorded before: "%s"')

        # Write.
        imageio.imwrite(maskpath, mask)
        return maskpath

    def close(self):
        pass


class MediaReader:
    '''
    A wrapper class around PictureReader and VideoReader.
    The purpose is to automatically understand if an image_id is a picture or video frame.
    If it is a picture, create PictureReader.
    If it is a video frame, create VideoReader.
    Reader can not later change, that is, can not mix pictures and video media.
    '''

    def __init__(self, rootdir):  # TODO: pass kwargs to self.reader.__init__
        self.rootdir = rootdir
        if not isinstance(rootdir, str):
            raise ValueError('rootdir must be a string, got %s' % str(rootdir))
        self.reader = None  # Lazy initialization.

        self.image_cache = {}  # cache of previously read image(s)
        self.mask_cache = {}  # cache of previously read mask(s)

    def close(self):
        if self.reader is not None:
            self.reader.close()

    def _lazy_init(self, full_image_id):
        if self.reader is not None:
            return
        media_type = _getMediaType(full_image_id)
        if media_type == 'PICTURE':
            self.reader = PictureReader()
        elif media_type == 'VIDEO':
            self.reader = VideoReader()
        else:
            assert False, '_getMediaType was supposed to have raised an error'

    def imread(self, image_id):
        logging.debug('Try to read image_id "%s" with rootdir "%s"', image_id,
                      self.rootdir)

        # Try to read from cache.
        if image_id in self.image_cache:
            logging.debug('imread: found image in cache')
            return self.image_cache[image_id]  # get cached image if possible

        full_image_id = op.join(self.rootdir, image_id)
        self._lazy_init(full_image_id)
        image = self.reader.imread(full_image_id)

        logging.debug('imread: new image, updating cache')
        self.image_cache = {
            image_id: image
        }  # currently only 1 image in the cache
        return image

    def maskread(self, mask_id):
        logging.debug('Try to read mask_id "%s" with rootdir "%s"', mask_id,
                      self.rootdir)

        # Try to read from cache.
        if mask_id in self.mask_cache:
            logging.debug('maskread: found mask in cache')
            return self.mask_cache[mask_id]  # get cached mask if possible

        full_mask_id = op.join(self.rootdir, mask_id)
        self._lazy_init(full_mask_id)
        mask = self.reader.imread(full_mask_id)

        logging.debug('maskread: new mask, updating cache')
        self.mask_cache = {
            mask_id: mask
        }  # currently only 1 image in the cache
        return mask

    def checkIdExists(self, image_or_mask_id):
        return self.reader.checkIdExists(
            op.join(self.rootdir, image_or_mask_id))

    def getHeightAndWidth(self, image_or_mask_id):
        full_image_or_mask_id = op.join(self.rootdir, image_or_mask_id)
        self._lazy_init(full_image_or_mask_id)
        return self.reader.getHeightAndWidth(full_image_or_mask_id)


class MockWriter:

    def __init__(self, imagedir=None, maskdir=None):
        # TODO: Use imagedir and maskdir to return proper paths.
        self.imagedir = imagedir  # Ignored.
        self.maskdir = maskdir  # Ignored.
        self.image_current_frame = -1
        self.mask_current_frame = -1

    def imwrite(self, image, namehint=None):
        if namehint is not None and not isinstance(namehint, str):
            raise ValueError('namehint must be a string, got %s' %
                             str(namehint))

        # If "namehint" is not specified, compute name as the next frame.
        if namehint is None:
            self.image_current_frame += 1
            name = self.image_current_frame
        else:
            name = op.basename(namehint)
        return name

    def maskwrite(self, mask, namehint=None):
        if namehint is not None and not isinstance(namehint, str):
            raise ValueError('namehint must be a string, got %s' %
                             str(namehint))

        # If "namehint" is not specified, compute name as the next frame.
        if namehint is None:
            self.mask_current_frame += 1
            name = self.mask_current_frame
        else:
            name = op.basename(namehint)
        return name

    def close(self):
        pass


class MediaWriter:
    '''
    A wrapper class around PictureWriter and VideoWriter. The purpose is
    1) to avoid if-else in the operation code, depending on how a user would
       like to record data.
    2) return paths relative to rootdir, if needed.
    '''

    def __init__(self,
                 media_type,
                 image_media=None,
                 mask_media=None,
                 rootdir='',
                 overwrite=False):
        '''
        Args:
          media_type:   "video" for multiple images in ffmpeg formats,
                        "pictures" for directory with single images in ffmpeg formats.
          image_media:  path for "imagefile" to video or directory, depending on media_type.
          mask_media:   path for "maskfile" to video or directory, depending on media_type.
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
            self.writer = VideoWriter(vimagefile=image_media,
                                      vmaskfile=mask_media,
                                      overwrite=overwrite)
        elif media_type == 'pictures':
            self.writer = PictureWriter(imagedir=image_media,
                                        maskdir=mask_media,
                                        overwrite=overwrite)
        elif media_type == 'mock':
            self.writer = MockWriter(imagedir=image_media, maskdir=mask_media)
        else:
            raise ValueError(
                '"media" must be either "video" or "pictures", not %s' %
                media_type)

    def imwrite(self, image, namehint=None):
        '''
        Args:
          namehint:  A file in directory will have basename(namehint), not a sequential number.
                 (Only for media_type="pictures".)
        '''
        if self.media_type == 'video':
            image_id = self.writer.imwrite(image)
        elif self.media_type == 'pictures':
            image_id = self.writer.imwrite(image, namehint=namehint)
        elif self.media_type == 'mock':
            image_id = self.writer.imwrite(image, namehint=namehint)
        else:
            assert False, "We should not be here."

        return op.relpath(image_id, self.rootdir)

    def maskwrite(self, mask, namehint=None):
        '''
        Args:
          namehint:  A file in directory will have basename(namehint), not a sequential number.
                 (Only for media_type="pictures".)
        '''
        if self.media_type == 'video':
            mask_id = self.writer.maskwrite(mask)
        elif self.media_type == 'pictures':
            mask_id = self.writer.maskwrite(mask, namehint=namehint)
        elif self.media_type == 'mock':
            mask_id = self.writer.maskwrite(mask, namehint=namehint)
        else:
            assert False, "We should not be here."

        return op.relpath(mask_id, self.rootdir)

    def close(self):
        self.writer.close()
