import os, os.path as op
import cv2  # Use a different library to test image loading.
import numpy as np
import shutil
import unittest
import tempfile
import progressbar
import nose

from shuffler.backend import backend_media


def _diff(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.abs(a - b).sum() / (np.abs(a) + np.abs(b)).sum()


class TestGetPictureSize(unittest.TestCase):

    def test_jpg(self):
        height, width = backend_media.getPictureSize(
            'testdata/cars/images/000000.jpg')
        self.assertEqual(width, 800)
        self.assertEqual(height, 700)

    def test_png(self):
        height, width = backend_media.getPictureSize(
            'testdata/cars/masks/000000.png')
        self.assertEqual(width, 800)
        self.assertEqual(height, 700)


class TestVideoReader(unittest.TestCase):

    def setUp(self):
        self.reader = backend_media.VideoReader()

    def tearDown(self):
        self.reader.close()

    def test_imread_same(self):
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = self.reader.imread('testdata/cars/images.avi/0')
        # This next one should be from cache.
        img = self.reader.imread('testdata/cars/images.avi/0')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.15)

    def test_imread_sequence(self):
        img = self.reader.imread('testdata/cars/images.avi/0')
        img = self.reader.imread('testdata/cars/images.avi/1')
        img = self.reader.imread('testdata/cars/images.avi/2')
        img_gt = cv2.imread('testdata/cars/images/000002.jpg')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.15)

    def test_imread_inverse_sequence(self):
        img = self.reader.imread('testdata/cars/images.avi/2')
        img = self.reader.imread('testdata/cars/images.avi/1')
        img = self.reader.imread('testdata/cars/images.avi/0')
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.15)

    def test_imread_two_videos(self):
        # First video.
        img1_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img1 = self.reader.imread('testdata/cars/images.avi/000000')
        self.assertEqual(img1.shape, img1_gt.shape)
        self.assertLess(_diff(img1, img1_gt), 0.15)
        # Second video.
        img2_gt = cv2.imread('testdata/moon/images/000000.jpg')
        img2 = self.reader.imread('testdata/moon/images.avi/0')
        self.assertEqual(img2.shape, img2_gt.shape)
        self.assertLess(_diff(img2, img2_gt), 0.25)
        # First video.
        img1_gt = cv2.imread('testdata/cars/images/000002.jpg')
        img1 = self.reader.imread('testdata/cars/images.avi/2')
        self.assertEqual(img1.shape, img1_gt.shape)
        self.assertLess(_diff(img1, img1_gt), 0.15)
        # Second video.
        img2_gt = cv2.imread('testdata/moon/images/000002.jpg')
        img2 = self.reader.imread('testdata/moon/images.avi/2')
        self.assertEqual(img2.shape, img2_gt.shape)
        self.assertLess(_diff(img2, img2_gt), 0.25)

    def test_imread_out_of_range_frameid(self):
        with self.assertRaises(IndexError):
            self.reader.imread('testdata/cars/images.avi/10')

    def test_imread_negative_frameid(self):
        with self.assertRaises(ValueError):
            self.reader.imread('testdata/cars/images.avi/-1')

    def test_imread_bad_frameid(self):
        with self.assertRaises(ValueError):
            self.reader.imread('testdata/cars/images.avi/dummy')

    def test_imread_non_exist(self):
        with self.assertRaises(ValueError):
            self.reader.imread('testdata/cars/images/dummy.avi/0')

    def test_maskread(self):
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = self.reader.maskread('testdata/cars/masks.avi/0')
        self.assertEqual(len(mask.shape), 2)
        self.assertEqual(mask.shape, mask_gt.shape)
        self.assertLess(_diff(mask, mask_gt), 0.01)

    def test_maskreadNonExist(self):
        with self.assertRaises(ValueError):
            self.reader.maskread('testdata/cars/dummy.avi/0')

    def test_maskreadBadImage(self):
        with self.assertRaises(Exception):
            self.reader.maskread('testdata/cars/masks.avi/dummy')


class TestVideoWriter(unittest.TestCase):

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    # TODO:
    #   test_imwrite_works should check quality.

    def test_init_create_files(self):
        image_media = op.join(self.work_dir, 'images.avi')
        mask_media = op.join(self.work_dir, 'masks.avi')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        self.assertTrue(op.exists(image_media))
        self.assertTrue(op.exists(mask_media))

    def test_init_no_extension(self):
        # Instead of "images.avi".
        image_media = op.join(self.work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media)
        with self.assertRaises(TypeError):
            writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()

    def test_init_without_overwrite(self):
        # For image.
        image_media = op.join(self.work_dir, 'images.avi')
        with open(image_media, 'w'):
            pass
        with self.assertRaises(ValueError):
            backend_media.MediaWriter(media_type='video',
                                      image_media=image_media)
        # For mask.
        mask_media = op.join(self.work_dir, 'masks.avi')
        with open(mask_media, 'w'):
            pass
        with self.assertRaises(ValueError):
            backend_media.MediaWriter(media_type='video',
                                      mask_media=mask_media)

    def test_init_overwrite_files(self):
        image_media = op.join(self.work_dir, 'images.avi')
        mask_media = op.join(self.work_dir, 'masks.avi')
        with open(image_media, 'w'):
            pass
        with open(mask_media, 'w'):
            pass
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media,
                                           overwrite=True)
        writer.close()
        self.assertTrue(op.exists(image_media))
        self.assertTrue(op.exists(mask_media))

    def test_imwrite_works(self):
        writer = backend_media.VideoWriter(
            vimagefile=op.join(self.work_dir, 'images.avi'))
        img = cv2.imread('testdata/cars/images/000000.jpg')
        writer.imwrite(img)
        img = cv2.imread('testdata/cars/images/000001.jpg')
        writer.imwrite(img)
        img = cv2.imread('testdata/cars/images/000002.jpg')
        writer.imwrite(img)
        writer.close()
        self.assertTrue(op.exists(op.join(self.work_dir, 'images.avi')))

    def test_maskwrite_works(self):
        writer = backend_media.VideoWriter(
            vmaskfile=op.join(self.work_dir, 'masks.avi'))
        mask = cv2.imread('testdata/cars/masks/000000.png')
        writer.maskwrite(mask)
        mask = cv2.imread('testdata/cars/masks/000001.png')
        writer.maskwrite(mask)
        mask = cv2.imread('testdata/cars/masks/000002.png')
        writer.maskwrite(mask)
        writer.close()


class TestPictureReader(unittest.TestCase):

    def setUp(self):
        self.reader = backend_media.PictureReader()

    def tearDown(self):
        self.reader.close()

    def test_imread(self):
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = self.reader.imread('testdata/cars/images/000000.jpg')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_imreadNonExist(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.imread('testdata/cars/images/dummy.jpg')

    def test_imreadBadImage(self):
        with self.assertRaises(Exception):
            self.reader.imread('testdata/cars/images/badImage.jpg')

    def test_maskread(self):
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = self.reader.maskread('testdata/cars/masks/000000.png')
        self.assertEqual(len(mask.shape), 2)
        self.assertEqual(mask.shape, mask_gt.shape)
        self.assertLess(_diff(mask, mask_gt), 0.02)

    def test_maskreadNonExist(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.maskread('testdata/cars/images/dummy.png')

    def test_maskreadBadImage(self):
        with self.assertRaises(Exception):
            self.reader.maskread('testdata/cars/images/badImage.jpg')


class TestPictureWriter(unittest.TestCase):

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_close_without_error(self):
        self.writer = backend_media.PictureWriter()
        self.writer.close()

    def test_init_create_dir(self):
        image_media = op.join(self.work_dir, 'images')
        mask_media = op.join(self.work_dir, 'masks')
        backend_media.MediaWriter(media_type='pictures',
                                  image_media=image_media,
                                  mask_media=mask_media)
        self.assertTrue(op.exists(image_media))
        self.assertTrue(op.exists(mask_media))

    def test_init_without_overwrite(self):
        # For image.
        image_media = op.join(self.work_dir, 'images')
        os.makedirs(image_media)
        with self.assertRaises(ValueError):
            backend_media.MediaWriter(media_type='pictures',
                                      image_media=image_media)
        # For mask.
        mask_media = op.join(self.work_dir, 'masks')
        os.makedirs(mask_media)
        with self.assertRaises(ValueError):
            backend_media.MediaWriter(media_type='pictures',
                                      mask_media=mask_media)

    def test_init_overwrite_dirs(self):
        image_media = op.join(self.work_dir, 'images')
        mask_media = op.join(self.work_dir, 'masks')
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
        self.assertTrue(op.exists(image_media))
        self.assertTrue(op.exists(mask_media))
        # Check that files dissapeared from the dirs.
        self.assertFalse(op.exists(op.join(image_media, 'myfile')))
        self.assertFalse(op.exists(op.join(mask_media, 'myfile')))

    def test_imwrite(self):
        img_path = op.join(self.work_dir, 'testdata/cars/images/0.jpg')
        self.writer = backend_media.PictureWriter(
            imagedir=op.dirname(img_path))
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        self.writer.imwrite(img_gt)
        img = cv2.imread(img_path)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_imwrite_namehint(self):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/myname.jpg.
        img_path = op.join(self.work_dir, 'images/myname.jpg')
        img_dir = op.dirname(img_path)
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='myname.jpg')
        # Check that the image is there and that it is the same as original.
        self.assertTrue(op.exists(img_path))
        img = cv2.imread(img_path)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_imwrite_namehint_noext(self):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/myname.jpg.
        img_path = op.join(self.work_dir, 'images/myname.jpg')
        img_dir = op.dirname(img_path)
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='myname')
        # Check that the image is there and that it is the same as original.
        self.assertTrue(op.exists(img_path))
        img = cv2.imread(img_path)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_imwrite_namehint_nested(self):
        # Load the image to be written.
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        # Image will be in <temp dir>/images/myname.jpg.
        img_path = op.join(self.work_dir, 'images/mydir/myname.jpg')
        img_dir = op.dirname(op.dirname(img_path))
        self.writer = backend_media.PictureWriter(imagedir=img_dir)
        # Workhorse.
        self.writer.imwrite(img_gt, namehint='mydir/myname.jpg')
        # Check that the image is there and that it is the same as original.
        self.assertTrue(op.exists(img_path))
        img = cv2.imread(img_path)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_imwrite_jpg_quality(self):
        img_path = op.join(self.work_dir, 'testdata/cars/images/0.jpg')
        self.writer = backend_media.PictureWriter(
            imagedir=op.dirname(img_path), jpg_quality=100)
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        self.writer.imwrite(img_gt)
        img = cv2.imread(img_path)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_maskwrite(self):
        mask_path = op.join(self.work_dir, 'testdata/cars/masks/0.png')
        self.writer = backend_media.PictureWriter(
            maskdir=op.dirname(mask_path))
        mask_gt = cv2.imread('testdata/cars/masks/000000.png')
        self.writer.maskwrite(mask_gt)
        mask = cv2.imread(mask_path)
        self.assertLess(_diff(mask, mask_gt), 0.0001)

    def test_maskwrite_namehint(self):
        mask_media = op.join(self.work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        mask_gt = cv2.imread('testdata/cars/masks/000000.png')
        writer.close()
        writer.maskwrite(mask_gt, namehint='mymask')
        mask = cv2.imread(op.join(mask_media, 'mymask.png'))
        self.assertLess(_diff(mask, mask_gt), 0.002)
        writer.close()


class TestMediaReader(unittest.TestCase):

    def test_closeUninitWithoutError(self):
        reader = backend_media.MediaReader(rootdir='.')
        reader.close()

    def test_closePictureWithoutError(self):
        reader = backend_media.MediaReader(rootdir='.')
        reader.imread('testdata/cars/images/000000.jpg')
        reader.close()

    def test_imreadPicture(self):
        reader = backend_media.MediaReader(rootdir='.')
        img_gt = cv2.imread('testdata/cars/images/000000.jpg')
        img = reader.imread('testdata/cars/images/000000.jpg')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.10)
        # Second read.
        img = reader.imread('testdata/cars/images/000000.jpg')
        self.assertEqual(img.shape, img_gt.shape)
        self.assertLess(_diff(img, img_gt), 0.10)

    def test_maskreadPicture(self):
        reader = backend_media.MediaReader(rootdir='.')
        mask_gt = cv2.imread('testdata/cars/masks/000000.png',
                             cv2.IMREAD_GRAYSCALE)
        mask = reader.maskread('testdata/cars/masks/000000.png')
        self.assertEqual(mask.shape, mask_gt.shape)
        self.assertLess(_diff(mask, mask_gt), 0.02)
        # Second read.
        mask = reader.imread('testdata/cars/masks/000000.png')
        self.assertEqual(mask.shape, mask_gt.shape)
        self.assertLess(_diff(mask, mask_gt), 0.02)

    def test_imreadNotExisting(self):
        reader = backend_media.MediaReader(rootdir='.')
        with self.assertRaises(TypeError):
            reader.imread('testdata/cars/images/noExisting.jpg')

    def test_imreadNotExistingRootdir(self):
        reader = backend_media.MediaReader(rootdir='testdata/cars')
        with self.assertRaises(TypeError):
            reader.imread('noExisting.jpg')

    def test_imreadNotPictureOrVideo(self):
        reader = backend_media.MediaReader(rootdir='.')
        with self.assertRaises(TypeError):
            reader.imread('testdata/cars/images/000004.txt')

    def test_imreadNotPictureOrVideoRootdir(self):
        reader = backend_media.MediaReader(rootdir='testdata/cars')
        with self.assertRaises(TypeError):
            reader.imread('images/000005.txt')


class TestMediaWriter(unittest.TestCase):
    ''' Test only that necessary files exists. Do not check image content. '''

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_close_uninit_without_error(self):
        writer = backend_media.MediaWriter(media_type='pictures')
        writer.close()
        writer = backend_media.MediaWriter(media_type='video')
        writer.close()

    def test_close_without_error(self):
        # Pictures.
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=op.join(
                                               self.work_dir, 'pictures'))
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()
        # Video.
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=op.join(
                                               self.work_dir, 'video.avi'))
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()

    def test_video_init_create_files(self):
        image_media = op.join(self.work_dir, 'images.avi')
        mask_media = op.join(self.work_dir, 'masks.avi')
        writer = backend_media.MediaWriter(media_type='video',
                                           image_media=image_media,
                                           mask_media=mask_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        self.assertTrue(op.exists(image_media))
        self.assertTrue(op.exists(mask_media))

    def test_pictures_imwrite(self):
        image_media = op.join(self.work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=image_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8))
        writer.close()
        self.assertTrue(op.exists(op.join(image_media, '0.jpg')))

    def test_pictures_imwrite_namehint(self):
        image_media = op.join(self.work_dir, 'images')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           image_media=image_media)
        writer.imwrite(np.zeros((100, 100, 3), dtype=np.uint8),
                       namehint='myimage')
        writer.close()
        self.assertTrue(op.exists(op.join(image_media, 'myimage.jpg')))

    def test_pictures_maskwrite(self):
        mask_media = op.join(self.work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8))
        writer.close()
        self.assertTrue(op.exists(op.join(mask_media, '0.png')))

    def test_pictures_maskwrite_namehint(self):
        mask_media = op.join(self.work_dir, 'masks')
        writer = backend_media.MediaWriter(media_type='pictures',
                                           mask_media=mask_media)
        writer.maskwrite(np.zeros((100, 100), dtype=np.uint8),
                         namehint='mymask')
        writer.close()
        self.assertTrue(op.exists(op.join(mask_media, 'mymask.png')))


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
