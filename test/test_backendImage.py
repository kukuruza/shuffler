import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random
import logging
import skimage.io  # Use a different library to test against skimage.io
import numpy as np
import unittest

from lib import backendImages


class TestPictureReader (unittest.TestCase):

  def setUp (self):
    self.reader = backendImages.PictureReader()

  def test_closeWithoutError(self):
    self.reader.close()

  def test_imread (self):
    img_gt = skimage.io.imread('images/000000.jpg')
    img = self.reader.imread('images/000000.jpg')
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())

    img_gt = skimage.io.imread('images/000000.jpg')
    img = self.reader.imread(os.path.abspath('images/000000.jpg'))
    self.assertEqual(img.shape, img_gt.shape)
    self.assertTrue((img == img_gt).all())

  def test_imreadNonExist (self):
    with self.assertRaises(FileNotFoundError):
      self.reader.imread('images/dummy.jpg')

  def test_imreadBadImage (self):
    with self.assertRaises(Exception):
      self.reader.imread('images/badImage.jpg')


  def test_maskread (self):
    mask_gt = skimage.io.imread('masks/000000.png', as_gray=True)
    mask = self.reader.maskread('masks/000000.png')
    self.assertEqual(len(mask.shape), 2)
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertTrue((mask == mask_gt).all())

    mask_gt = skimage.io.imread('masks/000000.png', as_gray=True)
    mask = self.reader.maskread(os.path.abspath('masks/000000.png'))
    self.assertEqual(len(mask.shape), 2)
    self.assertEqual(mask.shape, mask_gt.shape)
    self.assertTrue((mask == mask_gt).all())

  def test_maskreadNonExist (self):
    with self.assertRaises(FileNotFoundError):
      self.reader.maskread('images/dummy.png')

  def test_maskreadBadImage (self):
    with self.assertRaises(Exception):
      self.reader.maskread('images/badImage.jpg')



class TestPictureWriter (unittest.TestCase):

  def tearDown(self):
    if os.path.exists('images/000000.temp.jpg'):
      os.remove('images/000000.temp.jpg')
    if os.path.exists('masks/000000.temp.png'):
      os.remove('masks/000000.temp.png')

  def test_closeWithoutError(self):
    self.writer = backendImages.PictureWriter()
    self.writer.close()

  def _diff(self, a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.abs(a - b).sum() / (a + b).sum()

  def test_imwriteJpgQualityNotSpec (self):
    self.writer = backendImages.PictureWriter()
    img_gt = skimage.io.imread('images/000000.jpg')
    self.writer.imwrite('images/000000.temp.jpg', img_gt)
    img = skimage.io.imread('images/000000.temp.jpg')
    self.assertLess(self._diff(img, img_gt), 0.01)

  def test_imwriteJpgQuality (self):
    self.writer = backendImages.PictureWriter(jpg_quality=100)
    img_gt = skimage.io.imread('images/000000.jpg')
    self.writer.imwrite('images/000000.temp.jpg', img_gt)
    img = skimage.io.imread('images/000000.temp.jpg')
    self.assertLess(self._diff(img, img_gt), 0.001)

  def test_maskwrite (self):
    self.writer = backendImages.PictureWriter()
    mask_gt = skimage.io.imread('masks/000000.png')
    self.writer.maskwrite('masks/000000.temp.png', mask_gt)
    mask = skimage.io.imread('masks/000000.temp.png')
    self.assertLess(self._diff(mask, mask_gt), 0.01)


class TestGetImageSize(unittest.TestCase):

  def test_jpg(self):
    height, width = backendImages.getImageSize('images/000000.jpg')
    self.assertEqual(width, 800)
    self.assertEqual(height, 700)

  def test_png(self):
    height, width = backendImages.getImageSize('masks/000000.png')
    self.assertEqual(width, 800)
    self.assertEqual(height, 700)



if __name__ == '__main__':
  logging.basicConfig (level=logging.ERROR)
  unittest.main()
