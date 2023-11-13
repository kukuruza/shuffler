import os, os.path as op
import cv2  # Use a different library to test image loading.
import numpy as np
import shutil
import tempfile
import pytest

from shuffler.backend import backend_media


def _diff(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.abs(a - b).sum() / (np.abs(a) + np.abs(b)).sum()


class Test_GetPictureHeightAndWidth:
    def test_jpg(self):
        height, width = backend_media.getPictureHeightAndWidth(
            'testdata/cars/images/000000.jpg')
        assert width == 800
        assert height == 700

    def test_png(self):
        height, width = backend_media.getPictureHeightAndWidth(
            'testdata/cars/masks/000000.png')
        assert width == 800
        assert height == 700


class Test_GetMediaType:
    def test_picture(self):
        assert backend_media._getMediaType('a/b/c.jpg') == 'PICTURE'

    def test_video(self):
        assert backend_media._getMediaType('a/b/c.avi/0') == 'VIDEO'
        assert backend_media._getMediaType('a/b/c.avi/42') == 'VIDEO'

    def test_error(self):
        with pytest.raises(ValueError):
            backend_media._getMediaType('a/b/c/42')
        with pytest.raises(ValueError):
            backend_media._getMediaType('a/b/c.zebra/42')


class Test_VideoReader:
    @pytest.fixture()
    def reader(self):
        reader = backend_media.VideoReader()
        yield reader
        reader.close()

    def test_imread_same(self, reader):
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = reader.imread('testdata/cars/images.avi/0')
        # This next one should be from cache.
        img = reader.imread('testdata/cars/images.avi/0')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.15

    def test_imread_sequence(self, reader):
        img = reader.imread('testdata/cars/images.avi/0')
        img = reader.imread('testdata/cars/images.avi/1')
        img = reader.imread('testdata/cars/images.avi/2')
        img_gt = cv2.imread('testdata/cars/images/000002.jpg')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.15

    def test_imread_inverse_sequence(self, reader):
        img = reader.imread('testdata/cars/images.avi/2')
        img = reader.imread('testdata/cars/images.avi/1')
        img = reader.imread('testdata/cars/images.avi/0')
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.15

    def test_imread_two_videos(self, reader):
        # First video.
        img1_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img1 = reader.imread('testdata/cars/images.avi/000000')
        assert img1.shape == img1_gt.shape
        assert _diff(img1, img1_gt) < 0.15
        # Second video.
        img2_gt = cv2.imread('testdata/moon/images/000000.jpg')
        img2 = reader.imread('testdata/moon/images.avi/0')
        assert img2.shape == img2_gt.shape
        assert _diff(img2, img2_gt) < 0.25
        # First video.
        img1_gt = cv2.imread('testdata/cars/images/000002.jpg')
        img1 = reader.imread('testdata/cars/images.avi/2')
        assert img1.shape == img1_gt.shape
        assert _diff(img1, img1_gt) < 0.15
        # Second video.
        img2_gt = cv2.imread('testdata/moon/images/000002.jpg')
        img2 = reader.imread('testdata/moon/images.avi/2')
        assert img2.shape == img2_gt.shape
        assert _diff(img2, img2_gt) < 0.25

    def test_imread_out_of_range_frameid(self, reader):
        with pytest.raises(IndexError):
            reader.imread('testdata/cars/images.avi/10')

    def test_imread_negative_frameid(self, reader):
        with pytest.raises(IndexError):
            reader.imread('testdata/cars/images.avi/-1')

    def test_imread_bad_frameid(self, reader):
        with pytest.raises(ValueError):
            reader.imread('testdata/cars/images.avi/dummy')

    def test_imread_non_exist(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.imread('testdata/cars/images/dummy.avi/0')

    def test_maskread(self, reader):
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = reader.maskread('testdata/cars/masks.avi/0')
        assert len(mask.shape) == 2
        assert mask.shape == mask_gt.shape
        assert _diff(mask, mask_gt) < 0.01

    def test_maskread_non_exist(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.maskread('testdata/cars/dummy.avi/0')

    def test_maskread_bad_image(self, reader):
        with pytest.raises(Exception):
            reader.maskread('testdata/cars/masks.avi/dummy')

    def test_checkIdExists(self, reader):
        # First video.
        assert reader.checkIdExists('testdata/cars/images.avi/000')
        # Second video.
        assert reader.checkIdExists('testdata/moon/images.avi/0')
        # Something is off, e.g. video does not exist.
        assert not reader.checkIdExists('dummy.avi/0')

    def test_getHeightAndWidth(self, reader):
        # First video.
        height, width = reader.getHeightAndWidth(
            'testdata/cars/images.avi/000')
        assert width == 800
        assert height == 700
        # Second video.
        height, width = reader.getHeightAndWidth('testdata/moon/images.avi/0')
        assert width == 120
        assert height == 80
        # Something is off, e.g. video does not exist.
        with pytest.raises(Exception):
            reader.getHeightAndWidth('dummy')


class Test_VideoWriter:
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    # TODO:
    #   test_imwrite_works should check quality.

    def test_init_create_files(self, work_dir):
        image_media = op.join(work_dir, 'images.avi')
        mask_media = op.join(work_dir, 'masks.avi')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        assert op.exists(image_media)
        assert op.exists(mask_media)

    def test_init_no_extension(self, work_dir):
        # Instead of "images.avi".
        image_media = op.join(work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media)
        with pytest.raises(TypeError):
            writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()

    def test_init_without_overwrite(self, work_dir):
        # For image.
        image_media = op.join(work_dir, 'images.avi')
        with open(image_media, 'w'):
            pass
        with pytest.raises(ValueError):
            backend_media.MediaWriter(media_type='video',
                                      image_media=image_media)
        # For mask.
        mask_media = op.join(work_dir, 'masks.avi')
        with open(mask_media, 'w'):
            pass
        with pytest.raises(ValueError):
            backend_media.MediaWriter(media_type='video',
                                      mask_media=mask_media)

    def test_init_overwrite_files(self, work_dir):
        image_media = op.join(work_dir, 'images.avi')
        mask_media = op.join(work_dir, 'masks.avi')
        with open(image_media, 'w'):
            pass
        with open(mask_media, 'w'):
            pass
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media,
                                           overwrite=True)
        writer.close()
        assert op.exists(image_media)
        assert op.exists(mask_media)

    def test_imwrite_works(self, work_dir):
        writer = backend_media.VideoWriter(
            vimagefile=op.join(work_dir, 'images.avi'))
        img = cv2.imread('testdata/cars/images/000000.jpg')
        writer.imwrite(img)
        img = cv2.imread('testdata/cars/images/000001.jpg')
        writer.imwrite(img)
        img = cv2.imread('testdata/cars/images/000002.jpg')
        writer.imwrite(img)
        writer.close()
        assert op.exists(op.join(work_dir, 'images.avi'))

    def test_maskwrite_works(self, work_dir):
        writer = backend_media.VideoWriter(
            vmaskfile=op.join(work_dir, 'masks.avi'))
        mask = cv2.imread('testdata/cars/masks/000000.png')
        writer.maskwrite(mask)
        mask = cv2.imread('testdata/cars/masks/000001.png')
        writer.maskwrite(mask)
        mask = cv2.imread('testdata/cars/masks/000002.png')
        writer.maskwrite(mask)
        writer.close()


class Test_PictureReader:
    @pytest.fixture()
    def reader(self):
        reader = backend_media.PictureReader()
        yield reader
        reader.close()

    def test_imread(self, reader):
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = reader.imread('testdata/cars/images/000000.jpg')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.10

    def test_imread_non_exist(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.imread('testdata/cars/images/dummy.jpg')

    def test_imread_bad_image(self, reader):
        with pytest.raises(Exception):
            reader.imread('testdata/cars/images/badImage.jpg')

    def test_maskread(self, reader):
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = reader.maskread('testdata/cars/masks/000000.png')
        assert len(mask.shape) == 2
        assert mask.shape == mask_gt.shape
        assert _diff(mask, mask_gt) < 0.02

    def test_maskread_non_exist(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.maskread('testdata/cars/images/dummy.png')

    def test_maskread_bad_image(self, reader):
        with pytest.raises(Exception):
            reader.maskread('testdata/cars/images/badImage.jpg')

    def test_checkIdExists(self, reader):
        assert reader.checkIdExists('testdata/cars/images/000000.jpg')
        assert not reader.checkIdExists('dummy.jpg')

    def test_getHeightAndWidth(self, reader):
        height, width = reader.getHeightAndWidth(
            'testdata/cars/images/000000.jpg')
        assert width == 800
        assert height == 700
        # Something is off, e.g. the file does not exist.
        with pytest.raises(Exception):
            reader.getHeightAndWidth('dummy')


class Test_PictureWriter:
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def test_close_without_error(self):
        writer = backend_media.PictureWriter()
        writer.close()

    def test_init_create_dir(self, work_dir):
        image_media = op.join(work_dir, 'images')
        mask_media = op.join(work_dir, 'masks')
        backend_media.MediaWriter(media_type='pictures',
                                  image_media=image_media,
                                  mask_media=mask_media)
        assert op.exists(image_media)
        assert op.exists(mask_media)

    def test_init_without_overwrite(self, work_dir):
        # For image.
        image_media = op.join(work_dir, 'images')
        os.makedirs(image_media)
        with pytest.raises(ValueError):
            backend_media.MediaWriter(media_type='pictures',
                                      image_media=image_media)
        # For mask.
        mask_media = op.join(work_dir, 'masks')
        os.makedirs(mask_media)
        with pytest.raises(ValueError):
            backend_media.MediaWriter(media_type='pictures',
                                      mask_media=mask_media)

    def test_init_overwrite_dirs(self, work_dir):
        image_media = op.join(work_dir, 'images')
        mask_media = op.join(work_dir, 'masks')
        os.makedirs(image_media)
        os.makedirs(mask_media)
        with open(op.join(image_media, 'myfile'), 'w'):
            pass
        with open(op.join(mask_media, 'myfile'), 'w'):
            pass
        backend_media.MediaWriter(media_type='pictures',
                                  image_media=image_media,
                                  mask_media=mask_media,
                                  overwrite=True)
        # Check that dirs are recreated.
        assert op.exists(image_media)
        assert op.exists(mask_media)
        # Check that files dissapeared from the dirs.
        assert not op.exists(op.join(image_media, 'myfile'))
        assert not op.exists(op.join(mask_media, 'myfile'))

    def test_imwrite(self, work_dir):
        img_path = op.join(work_dir, 'testdata/cars/images/0.jpg')
        self.writer = backend_media.PictureWriter(
            imagedir=op.dirname(img_path))
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        self.writer.imwrite(img_gt)
        img = cv2.imread(img_path)
        assert _diff(img, img_gt) < 0.10

    def test_imwrite_namehint(self, work_dir):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/myname.jpg.
        img_path = op.join(work_dir, 'images/myname.jpg')
        img_dir = op.dirname(img_path)
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='myname.jpg')
        # Check that the image is there and that it is the same as original.
        assert op.exists(img_path)
        img = cv2.imread(img_path)
        assert _diff(img, img_gt) < 0.10

    def test_imwrite_namehint_noext(self, work_dir):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/myname.jpg.
        img_path = op.join(work_dir, 'images/myname.jpg')
        img_dir = op.dirname(img_path)
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='myname')
        # Check that the image is there and that it is the same as original.
        assert op.exists(img_path)
        img = cv2.imread(img_path)
        assert _diff(img, img_gt) < 0.10

    def test_imwrite_namehint_nested(self, work_dir):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/images/myname.jpg.
        img_path = op.join(work_dir, 'images/mydir/myname.jpg')
        img_dir = op.dirname(op.dirname(img_path))
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='mydir/myname.jpg')
        # Check that the image is there and that it is the same as original.
        assert op.exists(img_path)
        img = cv2.imread(img_path)
        assert _diff(img, img_gt) < 0.10

    def test_imwrite_jpg_quality(self, work_dir):
        img_path = op.join(work_dir, 'testdata/cars/images/0.jpg')
        self.writer = backend_media.PictureWriter(
            imagedir=op.dirname(img_path), jpg_quality=100)
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        self.writer.imwrite(img_gt)
        img = cv2.imread(img_path)
        assert _diff(img, img_gt) < 0.10

    def test_maskwrite(self, work_dir):
        mask_path = op.join(work_dir, 'testdata/cars/masks/0.png')
        self.writer = backend_media.PictureWriter(
            maskdir=op.dirname(mask_path))
        mask_gt = cv2.imread('testdata/cars/masks/000000.png')
        self.writer.maskwrite(mask_gt)
        mask = cv2.imread(mask_path)
        assert _diff(mask, mask_gt) < 0.0001

    def test_maskwrite_namehint(self, work_dir):
        mask_media = op.join(work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        mask_gt = cv2.imread('testdata/cars/masks/000000.png')
        writer.close()
        writer.maskwrite(mask_gt, namehint='mymask')
        mask = cv2.imread(op.join(mask_media, 'mymask.png'))
        assert _diff(mask, mask_gt) < 0.002
        writer.close()


class Test_MediaReader:
    @pytest.fixture()
    def reader(self):
        reader = backend_media.MediaReader(rootdir='.')
        yield reader
        reader.close()

    def test_imread_picture(self, reader):
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = reader.imread('testdata/cars/images/000000.jpg')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.10
        # Second read.
        img = reader.imread('testdata/cars/images/000000.jpg')
        assert img.shape == img_gt.shape
        assert _diff(img, img_gt) < 0.10

    def test_maskread_picture(self, reader):
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = reader.maskread('testdata/cars/masks/000000.png')
        assert mask.shape == mask_gt.shape
        assert _diff(mask, mask_gt) < 0.02
        # Second read.
        mask = reader.imread('testdata/cars/masks/000000.png')
        assert mask.shape == mask_gt.shape
        assert _diff(mask, mask_gt) < 0.02

    def test_imread_not_existing(self, reader):
        with pytest.raises(FileNotFoundError):
            reader.imread('testdata/cars/images/noExisting.jpg')

    def test_imread_not_existing_rootdir(self):
        reader = backend_media.MediaReader(rootdir='testdata/cars')
        with pytest.raises(FileNotFoundError):
            reader.imread('noExisting.jpg')

    def test_imread_not_picture_or_video(self, reader):
        with pytest.raises(ValueError):
            reader.imread('testdata/cars/images/000004.txt')

    def test_imread_not_picture_or_video_rootdir(self):
        reader = backend_media.MediaReader(rootdir='testdata/cars')
        with pytest.raises(ValueError):
            reader.imread('images/000005.txt')


class Test_MediaWriter:
    ''' Test only that necessary files exists. Do not check image content. '''
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def test_close_uninit_without_error(self):
        writer = backend_media.MediaWriter(media_type='pictures')
        writer.close()
        writer = backend_media.MediaWriter(media_type='video')
        writer.close()

    def test_close_without_error(self, work_dir):
        # Pictures.
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=op.join(
                                               work_dir, 'pictures'))
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()
        # Video.
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=op.join(
                                               work_dir, 'video.avi'))
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()

    def test_video_init_create_files(self, work_dir):
        image_media = op.join(work_dir, 'images.avi')
        mask_media = op.join(work_dir, 'masks.avi')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        assert op.exists(image_media)
        assert op.exists(mask_media)

    def test_pictures_imwrite(self, work_dir):
        image_media = op.join(work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=image_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()
        assert op.exists(op.join(image_media, '0.jpg'))

    def test_pictures_imwrite_namehint(self, work_dir):
        image_media = op.join(work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=image_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8),
                       namehint='myimage')
        writer.close()
        assert op.exists(op.join(image_media, 'myimage.jpg'))

    def test_pictures_maskwrite(self, work_dir):
        mask_media = op.join(work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        assert op.exists(op.join(mask_media, '0.png'))

    def test_pictures_maskwrite_namehint(self, work_dir):
        mask_media = op.join(work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8),
                         namehint='mymask')
        writer.close()
        assert op.exists(op.join(mask_media, 'mymask.png'))
