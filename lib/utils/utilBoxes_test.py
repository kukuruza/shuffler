import os, os.path as op
import logging
import sqlite3
import progressbar
import unittest
import pprint
import tempfile
import numpy as np
import cv2

from lib.utils import utilBoxes


class TestCropPatch(unittest.TestCase):
    @staticmethod
    def transformRoi(transform, roi):
        if not transform.shape == (3, 3):
            raise RuntimeError('transform must have shape (3, 3), not %s.' %
                               str(transform.shape))
        affine_transform = transform[:2]
        p1_old = np.array([roi[0], roi[1], 1])[:, np.newaxis]
        p2_old = np.array([roi[2], roi[3], 1])[:, np.newaxis]
        p1_new = affine_transform.dot(p1_old)
        p2_new = affine_transform.dot(p2_old)
        assert p1_new.shape == (2, 1), p1_new.shape
        assert p2_new.shape == (2, 1), p2_new.shape
        p1_new = p1_new[:, 0]
        p2_new = p2_new[:, 0]
        roi_new = [p1_new[0], p1_new[1], p2_new[0], p2_new[1]]
        return [int(round(x)) for x in roi_new]

    WIDTH = 200
    HEIGHT = 100

    @staticmethod
    def makeGradientImage(height,
                          width,
                          min_x=0,
                          min_y=0,
                          max_x=None,
                          max_y=None):
        '''
      Make an image of dimensions (height, width, 3) with the following channels:
         red:    vertical gradient [min_y, max_y), horizontally constant.
         green:  horizontal gradient [min_x, max_x), vertically constant.
         blue:   zeros.
      If max_x or max_y are not specified, they are computed such that
      the step of the gradient is 1 in that direction.
      '''
        if max_y is None:
            max_y = min_y + height
        if max_x is None:
            max_x = min_x + width
        step_y = (max_y - min_y) / float(height)
        step_x = (max_x - min_x) / float(width)
        red = np.arange(min_y, max_y,
                        step_y)[:, np.newaxis].dot(np.ones(
                            (1, width))).astype(dtype=np.uint8)
        green = np.ones(
            (height,
             1)).dot(np.arange(min_x, max_x,
                               step_x)[np.newaxis, :]).astype(dtype=np.uint8)
        blue = np.zeros((height, width), dtype=np.uint8)
        assert red.shape == (height, width), red.shape
        assert green.shape == (height, width), green.shape
        assert blue.shape == (height, width), blue.shape
        image = np.stack((red, green, blue), axis=2)
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8
        return image

    def setUp(self):
        self.image = TestCropPatch.makeGradientImage(TestCropPatch.HEIGHT,
                                                     TestCropPatch.WIDTH)

    def test_edgeOriginal_identity(self):
        roi = [0, 0, self.HEIGHT, self.WIDTH]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(
            self.HEIGHT, self.WIDTH)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = roi
        self.assertEqual(actual_roi, expected_roi)
        # Make sure transform matches.
        np.testing.assert_array_equal(transform, np.eye(3, 3, dtype=float))

    def test_edgeOriginal_outOfBoundary1(self):
        roi = [-10, 30, 10, 70]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(height=10,
                                                         width=40,
                                                         min_y=0,
                                                         min_x=30)
        expected_patch = np.pad(expected_patch, ((10, 0), (0, 0), (0, 0)))
        self.assertEqual(actual_patch.shape, expected_patch.shape)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeOriginal_outOfBoundary2(self):
        roi = [40, -10, 60, 30]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=30,
                                                         min_y=40,
                                                         min_x=0)
        expected_patch = np.pad(expected_patch, ((0, 0), (10, 0), (0, 0)))
        self.assertEqual(actual_patch.shape, expected_patch.shape)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeOriginal_outOfBoundary3(self):
        roi = [90, 30, 110, 70]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(height=10,
                                                         width=40,
                                                         min_y=90,
                                                         min_x=30)
        expected_patch = np.pad(expected_patch, ((0, 10), (0, 0), (0, 0)))
        self.assertEqual(actual_patch.shape, expected_patch.shape)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeOriginal_outOfBoundary4(self):
        roi = [40, 170, 60, 210]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=30,
                                                         min_y=40,
                                                         min_x=170)
        expected_patch = np.pad(expected_patch, ((0, 0), (0, 10), (0, 0)))
        self.assertEqual(actual_patch.shape, expected_patch.shape)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeOriginal(self):
        roi = [40, 30, 60, 70]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image, roi,
                                                      'original', None, None)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=40,
                                                         min_y=40,
                                                         min_x=30)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeConstant_targetSizeNone(self):
        roi = [40, 30, 60, 70]
        with self.assertRaises(RuntimeError):
            utilBoxes.cropPatch(self.image, roi, 'constant', None, None)

    def test_edgeConstant_noStretch(self):
        roi = [40, 30, 60, 70]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'constant',
                                                      target_height=40,
                                                      target_width=40)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=40,
                                                         min_y=40,
                                                         min_x=30)
        expected_patch = np.pad(expected_patch,
                                pad_width=((10, 10), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [10, 0, 30, 40]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeDistortY(self):
        roi = [45, 40, 55, 60]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'distort',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=20,
                                                         min_y=45,
                                                         min_x=40,
                                                         max_y=55)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 0, 20, 20]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundY_noStretch(self):
        roi = [45, 40, 55, 60]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=20,
                                                         min_y=40,
                                                         min_x=40)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundX_noStretch(self):
        roi = [40, 45, 60, 55]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=20,
                                                         min_y=40,
                                                         min_x=40)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundYBelow0_noStretch(self):
        roi = [0, 40, 10, 60]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=15,
                                                         width=20,
                                                         min_x=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((5, 0), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundXBelow0_noStretch(self):
        roi = [40, 0, 60, 10]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=15,
                                                         min_y=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 0), (5, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundYAboveTop_noStretch(self):
        roi = [90, 40, 100, 60]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=15,
                                                         width=20,
                                                         min_y=85,
                                                         min_x=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 5), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundXAboveTop_noStretch(self):
        roi = [40, 190, 60, 200]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=20,
                                                      target_width=20)
        expected_patch = TestCropPatch.makeGradientImage(height=20,
                                                         width=15,
                                                         min_y=40,
                                                         min_x=185)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 0), (0, 5), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        self.assertEqual(actual_roi, expected_roi)

    def test_edgeBackgroundYBelow0_stretch(self):
        roi = [0, 40, 10, 60]
        # Compare patches.
        actual_patch, transform = utilBoxes.cropPatch(self.image,
                                                      roi,
                                                      'background',
                                                      target_height=40,
                                                      target_width=40)
        expected_patch = TestCropPatch.makeGradientImage(height=30,
                                                         width=40,
                                                         min_y=0,
                                                         min_x=40,
                                                         max_y=15,
                                                         max_x=60)
        expected_patch = np.pad(expected_patch,
                                pad_width=((10, 0), (0, 0), (0, 0)),
                                mode='constant')
        norm = (actual_patch.mean() + expected_patch.mean()) / 2.
        actual_patch = actual_patch / norm
        expected_patch = expected_patch / norm
        np.testing.assert_array_almost_equal(actual_patch,
                                             expected_patch,
                                             decimal=0)
        # Compare roi.
        actual_roi = TestCropPatch.transformRoi(transform, roi)
        expected_roi = [10, 0, 30, 40]
        self.assertEqual(actual_roi, expected_roi)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    logging.basicConfig(level=logging.INFO)

    unittest.main()
