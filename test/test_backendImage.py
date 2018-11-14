import os, sys, os.path as op
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random
import logging
import skimage.io  # Use a different library to test image loading.
import numpy as np
import shutil
import unittest

from lib import backendImages


def _diff(a, b):
  a = a.astype(float)
  b = b.astype(float)
  return np.abs(a - b).sum() / (np.abs(a) + np.abs(b)).sum()


class TestGetPictureSize(unittest.TestCase):

  def test_jpg(self):
    height, width = backendImages.getPictureSize('cars/images/000000.jpg')
    self.assertEqual(width, 800)
    self.assertEqual(height, 700)

  def test_png(self):
    height, width = backendImages.getPictureSize('cars/masks/000000.png')
    self.assertEqual(width, 800)
    self.assertEqual(height, 700)



class TestVideoReader (unittest.TestCase):

  # TODO:
  #   test videos are closed after self.reader.close()

  def setUp (self):
    self.reader = backendImages.VideoReader(rootdir='.')

  def test_close_without_error(self):
    self.reader.close()

  def test_imread_same (self):
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img = self.reader.imread('cars/images/video.avi/000000')
    # This next one should be from cache.
    img = self.reader.imread('cars/images/video.avi/000000')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertLess(_diff(img, img_gt), 0.02)
    
  def test_imread_sequence (self):
    img = self.reader.imread('cars/images/video.avi/000000')
    img = self.reader.imread('cars/images/video.avi/000001')
    img = self.reader.imread('cars/images/video.avi/000002')
    img_gt = skimage.io.imread('cars/images/000002.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertLess(_diff(img, img_gt), 0.02)

  def test_imread_inverse_sequence (self):
    img = self.reader.imread('cars/images/video.avi/000002')
    img = self.reader.imread('cars/images/video.avi/000001')
    img = self.reader.imread('cars/images/video.avi/000000')
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertLess(_diff(img, img_gt), 0.05)

  def test_imread_two_videos (self):
    # First video.
    img1_gt = skimage.io.imread('cars/images/000000.jpg')
    img1 = self.reader.imread('cars/images/video.avi/000000')
    self.assertEqual(img1.shape, img1_gt.shape)
    self.assertLess(_diff(img1, img1_gt), 0.05)
    # Second video.
    img2_gt = skimage.io.imread('moon/images/000000.jpg')
    img2 = self.reader.imread('moon/images.avi/000000')
    self.assertEqual(img2.shape, img2_gt.shape)
    self.assertLess(_diff(img2, img2_gt), 0.05)
    # First video.
    img1_gt = skimage.io.imread('cars/images/000002.jpg')
    img1 = self.reader.imread('cars/images/video.avi/000002')
    self.assertEqual(img1.shape, img1_gt.shape)
    self.assertLess(_diff(img1, img1_gt), 0.05)
    # Second video.
    img2_gt = skimage.io.imread('moon/images/000002.jpg')
    img2 = self.reader.imread('moon/images.avi/000002')
    self.assertEqual(img2.shape, img2_gt.shape)
    self.assertLess(_diff(img2, img2_gt), 0.05)

  def test_imread_out_of_range_frameid (self):
    with self.assertRaises(ValueError):
      self.reader.imread('cars/images/video.avi/000010')

  def test_imread_negative_frameid (self):
    with self.assertRaises(ValueError):
      self.reader.imread('cars/images/video.avi/-000001')

  def test_imread_bad_frameid (self):
    with self.assertRaises(ValueError):
      self.reader.imread('cars/images/video.avi/dummy')

  def test_imread_non_exist (self):
    with self.assertRaises(ValueError):
      self.reader.imread('cars/images/dummy.avi/000000')

  def test_maskread (self):
    mask_gt = skimage.io.imread('cars/masks/000000.png', as_gray=True)
    mask = self.reader.maskread('cars/masks/video.avi/000000')
    self.assertEqual(len(mask.shape), 2)
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertLess(_diff(mask, mask_gt), 0.01)

  def test_maskreadNonExist (self):
    with self.assertRaises(ValueError):
      self.reader.maskread('cars/masks/dummy.avi/000000')

  def test_maskreadBadImage (self):
    with self.assertRaises(Exception):
      self.reader.maskread('cars/masks/video.avi/dummy')


class TestVideoWriter (unittest.TestCase):

  WORK_DIR = '/tmp/TestVideoWriter'

  # TODO:
  #   test_imwrite_works should check quality.

  def setUp (self):
    if not op.exists(self.WORK_DIR):
      os.makedirs(self.WORK_DIR)

  def tearDown(self):
    shutil.rmtree(self.WORK_DIR)

  def test_imwrite_works(self):
    writer = backendImages.VideoWriter(rootdir='.',
      vimagefile=op.join(self.WORK_DIR, 'images.avi'))
    img = skimage.io.imread('cars/images/000000.jpg')
    writer.imwrite(img)
    img = skimage.io.imread('cars/images/000001.jpg')
    writer.imwrite(img)
    img = skimage.io.imread('cars/images/000002.jpg')
    writer.imwrite(img)
    writer.close()
    self.assertTrue(op.exists(op.join(self.WORK_DIR, 'images.avi')))

  def test_maskwrite_works(self):
    writer = backendImages.VideoWriter(rootdir='.',
      vmaskfile=op.join(self.WORK_DIR, 'masks.avi'))
    mask = skimage.io.imread('cars/masks/000000.png')
    writer.maskwrite(mask)
    mask = skimage.io.imread('cars/masks/000001.png')
    writer.maskwrite(mask)
    mask = skimage.io.imread('cars/masks/000002.png')
    writer.maskwrite(mask)
    writer.close()

  def test_overwrite_flag(self):
    # Make a dummy file.
    with open(op.join(self.WORK_DIR, 'images.avi'), 'w') as f:
      f.write('existing data')
    # Create a writer with overwrite enabled.
    writer = backendImages.VideoWriter(rootdir='.',
      vimagefile=op.join(self.WORK_DIR, 'images.avi'),
      overwrite=True)
    # Make sure imwrite does not raise an exception.
    img = skimage.io.imread('cars/images/000000.jpg')
    writer.imwrite(img)
    writer.close()

  def test_no_overwrite_flag(self):
    # Make a dummy file.
    with open(op.join(self.WORK_DIR, 'images.avi'), 'w') as f:
      f.write('existing data')
    # Create a writer with overwrite enabled.
    writer = backendImages.VideoWriter(rootdir='.',
      vimagefile=op.join(self.WORK_DIR, 'images.avi'),
      overwrite=False)
    # Make sure imwrite does not raise an exception.
    img = skimage.io.imread('cars/images/000000.jpg')
    with self.assertRaises(FileExistsError):
      writer.imwrite(img)
      
  def test_create_directory(self):
    # Create a writer with vimagefile pointing to non-existent dir.
    writer = backendImages.VideoWriter(rootdir='.',
      vimagefile=op.join(self.WORK_DIR, 'not_existing/images.avi'))
    # Make sure imwrite does not raise an exception.
    img = skimage.io.imread('cars/images/000000.jpg')
    writer.imwrite(img)

  def test_rootdir(self):
    # Create a writer with a non-default rootdir.
    writer = backendImages.VideoWriter(rootdir=self.WORK_DIR,
      vimagefile='images.avi')
    # Make sure imwrite does not raise an exception.
    img = skimage.io.imread('cars/images/000000.jpg')
    writer.imwrite(img)
    writer.close()
    self.assertTrue(op.exists(op.join(self.WORK_DIR, 'images.avi')))



class TestPictureReader (unittest.TestCase):

  def setUp (self):
    self.reader = backendImages.PictureReader(rootdir='.')

  def test_closeWithoutError(self):
    self.reader.close()

  def test_imread (self):
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img = self.reader.imread('cars/images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())

  def test_imreadNonExist (self):
    with self.assertRaises(ValueError):
      self.reader.imread('cars/images/dummy.jpg')

  def test_imreadBadImage (self):
    with self.assertRaises(Exception):
      self.reader.imread('cars/images/badImage.jpg')


  def test_maskread (self):
    mask_gt = skimage.io.imread('cars/masks/000000.png', as_gray=True)
    mask = self.reader.maskread('cars/masks/000000.png')
    self.assertEqual(len(mask.shape), 2)
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertLess(_diff(mask, mask_gt), 0.02)
    #self.assertTrue((mask == mask_gt).all())

  def test_maskreadNonExist (self):
    with self.assertRaises(ValueError):
      self.reader.maskread('cars/images/dummy.png')

  def test_maskreadBadImage (self):
    with self.assertRaises(Exception):
      self.reader.maskread('cars/images/badImage.jpg')



class TestPictureWriter (unittest.TestCase):

  WORK_DIR = '/tmp/TestPictureWriter'

  def setUp (self):
    if not op.exists(self.WORK_DIR):
      os.makedirs(self.WORK_DIR)

  def tearDown(self):
    shutil.rmtree(self.WORK_DIR)

  def test_closeWithoutError(self):
    self.writer = backendImages.PictureWriter()
    self.writer.close()

  def test_imwriteJpgQualityNotSpec (self):
    self.writer = backendImages.PictureWriter()
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img_path = op.join(self.WORK_DIR, 'cars/images/000000.jpg')
    self.writer.imwrite(img_path, img_gt)
    img = skimage.io.imread(img_path)
    self.assertLess(_diff(img, img_gt), 0.01)

  def test_imwriteJpgQuality (self):
    self.writer = backendImages.PictureWriter(jpg_quality=100)
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img_path = op.join(self.WORK_DIR, 'cars/images/000000.jpg')
    self.writer.imwrite(img_path, img_gt)
    img = skimage.io.imread(img_path)
    self.assertLess(_diff(img, img_gt), 0.001)

  def test_maskwrite (self):
    self.writer = backendImages.PictureWriter()
    mask_gt = skimage.io.imread('cars/masks/000000.png')
    mask_path = op.join(self.WORK_DIR, 'cars/images/000000.jpg')
    self.writer.maskwrite(mask_path, mask_gt)
    mask = skimage.io.imread(mask_path)
    self.assertLess(_diff(mask, mask_gt), 0.01)

  def test_rootdir(self):
    # Create a writer with vimagefile pointing to non-existent dir.
    writer = backendImages.PictureWriter(rootdir=op.join(self.WORK_DIR))
    # Make sure imwrite does not raise an exception.
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    writer.imwrite('cars/images/000000.jpg', img_gt)
    self.assertTrue(op.exists(op.join(self.WORK_DIR, 'cars/images/000000.jpg')))



class TestImreader (unittest.TestCase):

  def test_closeUninitWithoutError(self):
    reader = backendImages.ImageryReader(rootdir='.')
    reader.close()    

  def test_closePictureWithoutError(self):
    reader = backendImages.ImageryReader(rootdir='.')
    img = reader.imread('cars/images/000000.jpg')
    reader.close()    

  def test_imreadPicture (self):
    reader = backendImages.ImageryReader(rootdir='.')
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img = reader.imread('cars/images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())
    # Second read.
    img = reader.imread('cars/images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())

  def test_imreadPictureRootdir (self):
    reader = backendImages.ImageryReader(rootdir='cars')
    img_gt = skimage.io.imread('cars/images/000000.jpg')
    img = reader.imread('images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())

  def test_maskreadPicture (self):
    reader = backendImages.ImageryReader(rootdir='.')
    mask_gt = skimage.io.imread('cars/masks/000000.png')
    mask = reader.maskread('cars/masks/000000.png')
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertLess(_diff(mask, mask_gt), 0.02)
    #self.assertTrue((mask == mask_gt).all())
    # Second read.
    mask = reader.imread('cars/masks/000000.png')
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertLess(_diff(mask, mask_gt), 0.02)
    #self.assertTrue((mask == mask_gt).all())

  def test_maskreadPictureRootdir (self):
    reader = backendImages.ImageryReader(rootdir='cars')
    mask_gt = skimage.io.imread('cars/masks/000000.png')
    mask = reader.maskread('masks/000000.png')
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertLess(_diff(mask, mask_gt), 0.02)
    #self.assertTrue((mask == mask_gt).all())

  def test_imreadNotExisting (self):
    reader = backendImages.ImageryReader(rootdir='.')
    with self.assertRaises(TypeError):
      reader.imread('cars/images/noExisting.jpg')

  def test_imreadNotExistingRootdir (self):
    reader = backendImages.ImageryReader(rootdir='cars')
    with self.assertRaises(TypeError):
      reader.imread('noExisting.jpg')

  def test_imreadNotPictureOrVideo (self):
    reader = backendImages.ImageryReader(rootdir='.')
    with self.assertRaises(TypeError):
      reader.imread('cars/images/000004.txt')

  def test_imreadNotPictureOrVideoRootdir (self):
    reader = backendImages.ImageryReader(rootdir='cars')
    with self.assertRaises(TypeError):
      reader.imread('images/000005.txt')




if __name__ == '__main__':
  logging.basicConfig (level=logging.ERROR)
  unittest.main()
