import numpy as np
import imageio
import pytest

from shuffler.utils import draw as draw_utils


class TestDrawFilledRoi:
    RED = (255, 0, 0)

    def test_regular_fullfill(self):
        ''' fill_opacity=1 must fill the area completely. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        expected_image = image.copy()
        expected_image[50:150, 100:200] = self.RED

        draw_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                  self.RED,
                                  fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_float_roi(self):
        ''' Test that it does not break for non-integer ROI. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        draw_utils._drawFilledRoi(image, (50.3, 100.7, 150.1, 200),
                                  self.RED,
                                  fill_opacity=1)

    def test_regular_nofill(self):
        ''' fill_opacity=0 must have no effect on the image. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        expected_image = image.copy()
        # Do NOT change the image, because fill_opacity=0.

        draw_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                  self.RED,
                                  fill_opacity=0)
        # np.testing.assert_array_equal(image, expected_image)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_grayscale_fullfill(self):
        ''' Grayscale images should be processed correctly. '''
        image = imageio.v2.imread('imageio:camera.png')
        assert len(image.shape) == 2, 'camera.png was expected to be grayscale'
        expected_image = image.copy()
        expected_image[50:150, 100:200] = 255

        draw_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                  255,
                                  fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary1_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.v2.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[0:150, 200:451] = self.RED

        draw_utils._drawFilledRoi(image, (-100, 200, 150, 600),
                                  self.RED,
                                  fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary2_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.v2.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[150:300, 0:200] = self.RED

        draw_utils._drawFilledRoi(image, (150, -100, 400, 200),
                                  self.RED,
                                  fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)


class TestDrawFilledPolygon:
    RED = (255, 0, 0)

    def test_regular_fullfill(self):
        ''' fill_opacity=1 must fill the area completely. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        expected_image = image.copy()
        expected_image[50:150, 100:300] = self.RED

        draw_utils._drawFilledPolygon(image, [(50, 100), (50, 300), (150, 300),
                                              (150, 100)],
                                      self.RED,
                                      fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_float_polygon(self):
        ''' Test that it does not break for non-integer polygon. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        draw_utils._drawFilledPolygon(image, [(50.1, 100.2), (50.3, 300.4),
                                              (150.5, 300.6), (150, 100)],
                                      self.RED,
                                      fill_opacity=1)

    def test_invalid_polygon(self):
        ''' Bad polygons must be quietly ignored. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        expected_image = image.copy()

        draw_utils._drawFilledPolygon(image, [(50, 100), (50, 200)],
                                      self.RED,
                                      fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_regular_nofill(self):
        ''' fill_opacity=0 must have no effect on the image. '''
        image = imageio.v2.imread('imageio:chelsea.png')
        expected_image = image.copy()
        # Do NOT change the image, because fill_opacity=0.

        draw_utils._drawFilledPolygon(image, [(50, 100), (50, 200), (150, 200),
                                              (150, 100)],
                                      self.RED,
                                      fill_opacity=0)
        # np.testing.assert_array_equal(image, expected_image)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_grayscale_fullfill(self):
        ''' Grayscale images should be processed correctly. '''
        image = imageio.v2.imread('imageio:camera.png')
        assert len(image.shape) == 2, 'camera.png was expected to be grayscale'
        expected_image = image.copy()
        expected_image[50:150, 100:200] = 255

        draw_utils._drawFilledPolygon(image, [(50, 100), (50, 200), (150, 200),
                                              (150, 100)],
                                      255,
                                      fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary1_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.v2.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[0:150, 200:451] = self.RED

        draw_utils._drawFilledPolygon(image, [(-100, 200), (-100, 600),
                                              (150, 600), (150, 200)],
                                      self.RED,
                                      fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary2_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.v2.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[150:350, 0:200] = self.RED

        draw_utils._drawFilledPolygon(image, [(150, -100), (150, 200),
                                              (500, 200), (500, -100)],
                                      self.RED,
                                      fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)
