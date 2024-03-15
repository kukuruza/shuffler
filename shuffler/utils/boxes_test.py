import pytest
import numpy as np
import collections

from shuffler.utils import boxes as boxes_utils


class Test_ValidateBbox:
    def test_ok(self):
        boxes_utils.validateBbox([1, 2, 3, 4])
        boxes_utils.validateBbox([1., 2., 3., 4.])
        boxes_utils.validateBbox((1, 2, 3, 4))
        boxes_utils.validateBbox(np.array([1, 2, 3, 4]))

    def test_not_sequence(self):
        with pytest.raises(TypeError):
            boxes_utils.validateBbox(42)

    def test_less_than_four_numbers(self):
        with pytest.raises(ValueError):
            boxes_utils.validateBbox([42])

    def test_more_than_four_numbers(self):
        with pytest.raises(ValueError):
            boxes_utils.validateBbox([1, 2, 3, 4, 5])

    def test_not_numbers(self):
        with pytest.raises(TypeError):
            boxes_utils.validateBbox([1, 2, 3, 'd'])

    def test_negative_dims(self):
        with pytest.raises(ValueError):
            boxes_utils.validateBbox([1, 2, 3, -1])


class Test_ValidateRoi:
    def test_ok(self):
        boxes_utils.validateRoi([1, 2, 3, 4])
        boxes_utils.validateRoi([1., 2., 3., 4.])
        boxes_utils.validateRoi((1, 2, 3, 4))
        boxes_utils.validateRoi(np.array([1, 2, 3, 4]))

    def test_not_sequence(self):
        with pytest.raises(TypeError):
            boxes_utils.validateRoi(42)

    def test_less_than_four_numbers(self):
        with pytest.raises(ValueError):
            boxes_utils.validateRoi([42])

    def test_more_than_four_numbers(self):
        with pytest.raises(ValueError):
            boxes_utils.validateRoi([1, 2, 3, 4, 5])

    def test_not_numbers(self):
        with pytest.raises(TypeError):
            boxes_utils.validateRoi([1, 2, 3, 'd'])

    def test_negativeDims(self):
        with pytest.raises(ValueError):
            boxes_utils.validateRoi([1, 2, 3, 1])  # negative width.
            boxes_utils.validateRoi([1, 2, 0, 4])  # negative height.


class Test_ValidatePolygon:
    def test_ok(self):
        boxes_utils.validatePolygon([(1, 1), (1, 2), (1, 3)])
        boxes_utils.validatePolygon([(1., 1.), (1., 2.), (1., 3.)])

        # Different types of iterable.
        boxes_utils.validatePolygon(((1, 1), (1, 2), (1, 3)))
        boxes_utils.validatePolygon([[1, 1], [1, 2], [1, 3]])
        boxes_utils.validatePolygon(np.array([[1, 1], [1, 2], [1, 3]]))

    def test_not_iterable(self):
        with pytest.raises(TypeError):
            boxes_utils.validatePolygon("abc")

    def test_not_numbers(self):
        with pytest.raises(TypeError):
            boxes_utils.validatePolygon([(1, 1), ('a', 2), (3, 4)])
            boxes_utils.validatePolygon([(1, 1), (2, 'b'), (3, 4)])

    def test_less_than_three_numbers(self):
        with pytest.raises(ValueError):
            boxes_utils.validatePolygon([(1, 1), (1, 2)])


class Test_Bbox2roi:
    def test_normal(self):
        assert boxes_utils.bbox2roi([1, 2, 3, 4]) == [2, 1, 6, 4]
        assert boxes_utils.bbox2roi((1, 2, 3, 4)) == [2, 1, 6, 4]

    def test_zero_dims(self):
        assert boxes_utils.bbox2roi([1, 2, 0, 0]) == [2, 1, 2, 1]


class Test_Roi2Bbox:
    def test_normal(self):
        assert boxes_utils.roi2bbox([2, 1, 6, 4]) == [1, 2, 3, 4]
        assert boxes_utils.roi2bbox((2, 1, 6, 4)) == [1, 2, 3, 4]

    def test_zero_dims(self):
        assert boxes_utils.roi2bbox([2, 1, 2, 1]) == [1, 2, 0, 0]


class Test_GetIoURoi:
    def test_identical(self):
        # Integer input.
        assert boxes_utils.getIoURoi([1, 2, 3, 4], [1, 2, 3, 4]) == 1.
        # Float input.
        assert boxes_utils.getIoURoi([10., 20., 30., 40.],
                                     [10., 20., 30., 40.]) == 1.

    def test_one_inside_the_other(self):
        # One is 50% width of the other and is completely inside.
        assert boxes_utils.getIoURoi([1, 2, 3, 4], [2, 2, 3, 4]) == 0.50

    def test_partial_overlap(self):
        # Overlap 33.3% perc on Y axis, identical area.
        assert boxes_utils.getIoURoi([1, 2, 4, 4], [3, 2, 6, 4]) == 0.20
        # Overlap 33.3% perc on X axis, identical area.
        assert boxes_utils.getIoURoi([2, 1, 4, 4], [2, 3, 4, 6]) == 0.20

    def test_no_overlap(self):
        # No overlap on Y axis, identical area.
        assert boxes_utils.getIoURoi([1, 2, 3, 4], [3, 2, 5, 4]) == 0.0
        # No overlap on X axis, identical area.
        assert boxes_utils.getIoURoi([2, 1, 4, 3], [2, 3, 4, 5]) == 0.0


class Test_GetIoUPolygon:
    def test_identical(self):
        assert boxes_utils.getIoUPolygon([(0, 0), (1, 1), (1, 0)],
                                         [(0, 0), (1, 1), (1, 0)]) == 1.

    def test_one_inside_the_other(self):
        # One is 50% larger than the other and is completely inside.
        assert boxes_utils.getIoUPolygon([(0, 0), (1, 1), (1, 0)],
                                         [(0, 0), (2, 2), (2, 0)]) == 0.25

    def test_partial_overlap(self):
        one_third = pytest.approx(1 / 3, 0.00001)
        assert boxes_utils.getIoUPolygon([(0, 0), (1, 1), (1, 0)],
                                         [(0, 1), (1, 0), (1, 1)]) == one_third

    def test_no_overlap(self):
        assert boxes_utils.getIoUPolygon([(0, 0), (1, 1), (1, 0)],
                                         [(5, 6), (6, 5), (6, 6)]) == 0.0

    def test_touch_in_one_point(self):
        assert boxes_utils.getIoUPolygon([(0, 0), (0, 1), (1, 1), (1, 0)],
                                         [(1, 1), (1, 2), (2, 2), (2, 1)]) == 0

    def test_touch_across_line_segment(self):
        assert boxes_utils.getIoUPolygon([(0, 0), (0, 1), (1, 1), (1, 0)],
                                         [(0, 1), (0, 2), (1, 2), (1, 1)]) == 0


class Test_ClipPolygonToRoi:
    def polygons_equivalent(self, yxs1, yxs2):
        deque1 = collections.deque(yxs1)
        deque2 = collections.deque(yxs2)
        assert any([
            deque1 == deque2 for _ in range(len(deque1))
            if not deque2.rotate()
        ]), (deque1, deque2)

    def test_identical(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [0, 0, 1, 1]
        assert boxes_utils.clipPolygonToRoi(yxs, roi) == yxs

    def test_polygon_inside_roi(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [-1, -1, 2, 2]
        assert boxes_utils.clipPolygonToRoi(yxs, roi) == yxs

    def test_partial_overlap(self):
        yxs = [(0, 0), (0, 3), (3, 0)]
        roi = [0, 0, 2, 2]
        self.polygons_equivalent(boxes_utils.clipPolygonToRoi(yxs, roi),
                                 [(0, 0), (0, 2), (2, 0)])

    def test_no_overlap(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [2, 2, 3, 3]
        assert boxes_utils.clipPolygonToRoi(yxs, roi) == []

    def test_touch_across_line_segment(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [1, 1, 2, 2]
        assert boxes_utils.clipPolygonToRoi(yxs, roi) == []

    def test_touch_across_one_line(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [1, 0, 2, 1]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == []


class Test_IntersectPolygonAndRoi:
    def polygons_equivalent(self, yxs1, yxs2):
        deque1 = collections.deque(yxs1)
        deque2 = collections.deque(yxs2)
        assert any([
            deque1 == deque2 for _ in range(len(deque1))
            if not deque2.rotate()
        ]), (deque1, deque2)

    def test_identical(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [0, 0, 1, 1]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == yxs

    def test_polygon_inside_roi(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [-1, -1, 2, 2]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == yxs

    def test_partial_overlap(self):
        yxs = [(0, 0), (0, 3), (3, 0)]
        roi = [0, 0, 2, 2]
        self.polygons_equivalent(boxes_utils.intersectPolygonAndRoi(yxs, roi),
                                 [(0, 0), (0, 2), (1, 2), (2, 1), (2, 0)])

    def test_no_overlap(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [2, 2, 3, 3]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == []

    def test_touch_across_line_segment(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [1, 1, 2, 2]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == []

    def test_touch_across_one_line(self):
        yxs = [(0, 0), (0, 1), (1, 1), (1, 0)]
        roi = [1, 0, 2, 1]
        assert boxes_utils.intersectPolygonAndRoi(yxs, roi) == []


class Test_ExpandRoi:
    def test_identity(self):
        roi = [0.5, 0.5, 100.5, 200.5]
        np.testing.assert_array_equal(
            np.array(roi), np.array(boxes_utils.expandRoi(roi, (0, 0))))

    def test_x_is_greater(self):
        roi = [0.5, 0.5, 100.5, 200.5]
        perc = (0.5, 1)
        expected = [-24.5, -99.5, 125.5, 300.5]
        np.testing.assert_array_equal(
            np.array(expected), np.array(boxes_utils.expandRoi(roi, perc)))

    def test_y_is_greater(self):
        roi = [0.5, 0.5, 100.5, 200.5]
        perc = (1, 0.5)
        expected = [-49.5, -49.5, 150.5, 250.5]
        np.testing.assert_array_equal(
            np.array(expected), np.array(boxes_utils.expandRoi(roi, perc)))

    def test_invalid(self):
        with pytest.raises(ValueError):
            boxes_utils.expandRoi([0.5, 0.5, 100.5, 200.5], (-0.8, 0))


class Test_ExpandPolygon:
    def test_identity(self):
        ys = [0.5, 0.5, 100.5]
        xs = [0.5, 200.5, 0.5]
        np.testing.assert_array_equal(
            np.array([ys, xs]),
            np.array(boxes_utils.expandPolygon(ys, xs, (0, 0))))

    def test_x_is_greater(self):
        ys = [0.5, 0.5, 100.5]
        xs = [0.5, 200.5, 0.5]
        perc = (0.5, 1)
        expected = ([-24.5, -24.5, 125.5], [-99.5, 300.5, -99.5])
        np.testing.assert_array_equal(
            np.array(expected),
            np.array(boxes_utils.expandPolygon(ys, xs, perc)))

    def test_y_is_greater(self):
        ys = [0.5, 0.5, 100.5]
        xs = [0.5, 200.5, 0.5]
        perc = (1, 0.5)
        expected = ([-49.5, -49.5, 150.5], [-49.5, 250.5, -49.5])
        np.testing.assert_array_equal(
            np.array(expected),
            np.array(boxes_utils.expandPolygon(ys, xs, perc)))

    def test_invalid(self):
        ys = [0.5, 0.5, 150.5]
        xs = [0.5, 300.5, 0.5]
        perc = (-0.8, 0)
        with pytest.raises(ValueError):
            boxes_utils.expandPolygon(ys, xs, perc)


class Test_ExpandRoiUpToRatio:
    def test_identity(self):
        roi = [0.5, 0.5, 100.5, 100.5]
        ratio = 1.
        expected_roi = roi
        expected_perc = 0
        actual_roi, actual_perc = boxes_utils.expandRoiUpToRatio(roi, ratio)
        np.testing.assert_array_equal(expected_roi, np.array(actual_roi))
        assert actual_perc == expected_perc

    def test_equal(self):
        roi = [0.5, 0.5, 100.5, 200.5]
        ratio = 1.
        expected_roi = [-49.5, 0.5, 150.5, 200.5]
        expected_perc = 1
        actual_roi, actual_perc = boxes_utils.expandRoiUpToRatio(roi, ratio)
        np.testing.assert_array_equal(expected_roi, np.array(actual_roi))
        assert actual_perc == expected_perc


class Test_ExpandPolygonUpToRatio:
    def test_identity(self):
        ys = [0.5, 0.5, 100.5]
        xs = [0.5, 100.5, 0.5]
        ratio = 1.
        expected_ys, expected_xs = ys, xs
        expected_perc = 0
        actual_ys, actual_xs, actual_perc = boxes_utils.expandPolygonUpToRatio(
            ys, xs, ratio)
        np.testing.assert_array_equal(expected_ys, np.array(actual_ys))
        np.testing.assert_array_equal(expected_xs, np.array(actual_xs))
        assert actual_perc == expected_perc

    def test_equal(self):
        ys = [0.5, 0.5, 100.5]
        xs = [0.5, 200.5, 0.5]
        ratio = 1.
        expected_ys, expected_xs = [-49.5, -49.5, 150.5], [0.5, 200.5, 0.5]
        expected_perc = 1
        actual_ys, actual_xs, actual_perc = boxes_utils.expandPolygonUpToRatio(
            ys, xs, ratio)
        np.testing.assert_array_equal(expected_ys, np.array(actual_ys))
        np.testing.assert_array_equal(expected_xs, np.array(actual_xs))
        assert actual_perc == expected_perc


class Test_CropPatch:
    @staticmethod
    def _transform_roi(transform, roi):
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
    def _make_gradient_mage(height,
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

    @pytest.fixture()
    def image(self):
        yield Test_CropPatch._make_gradient_mage(Test_CropPatch.HEIGHT,
                                                 Test_CropPatch.WIDTH)

    def test_edge_original_identity(self, image):
        roi = [0, 0, self.HEIGHT, self.WIDTH]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = image
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = roi
        assert actual_roi == expected_roi
        # Make sure transform matches.
        np.testing.assert_array_equal(transform, np.eye(3, 3, dtype=float))

    def test_edge_original_out_of_boundary1(self, image):
        roi = [-10, 30, 10, 70]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = Test_CropPatch._make_gradient_mage(height=10,
                                                            width=40,
                                                            min_y=0,
                                                            min_x=30)
        expected_patch = np.pad(expected_patch, ((10, 0), (0, 0), (0, 0)))
        assert actual_patch.shape == expected_patch.shape
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        assert actual_roi == expected_roi

    def test_edge_original_out_of_boundary2(self, image):
        roi = [40, -10, 60, 30]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=30,
                                                            min_y=40,
                                                            min_x=0)
        expected_patch = np.pad(expected_patch, ((0, 0), (10, 0), (0, 0)))
        assert actual_patch.shape == expected_patch.shape
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        assert actual_roi == expected_roi

    def test_edge_original_out_of_boundary3(self, image):
        roi = [90, 30, 110, 70]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = Test_CropPatch._make_gradient_mage(height=10,
                                                            width=40,
                                                            min_y=90,
                                                            min_x=30)
        expected_patch = np.pad(expected_patch, ((0, 10), (0, 0), (0, 0)))
        assert actual_patch.shape == expected_patch.shape
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        assert actual_roi == expected_roi

    def test_edge_original_out_of_boundary4(self, image):
        roi = [40, 170, 60, 210]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=30,
                                                            min_y=40,
                                                            min_x=170)
        expected_patch = np.pad(expected_patch, ((0, 0), (0, 10), (0, 0)))
        assert actual_patch.shape == expected_patch.shape
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        assert actual_roi == expected_roi

    def test_edge_original(self, image):
        roi = [40, 30, 60, 70]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(
            image, roi, 'original', None, None)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=40,
                                                            min_y=40,
                                                            min_x=30)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 40]
        assert actual_roi == expected_roi

    def test_edge_constant_target_size_none(self, image):
        roi = [40, 30, 60, 70]
        with pytest.raises(RuntimeError):
            boxes_utils.cropPatch(image, roi, 'constant', None, None)

    def test_edge_constant_less_than_two_integer_pixels(self, image):
        roi = [9.1, 20, 11.9, 40]
        with pytest.raises(ValueError):
            boxes_utils.cropPatch(image, roi, 'constant', 20, 20)

    def test_edgeConstant_noStretch(self, image):
        roi = [40, 30, 60, 70]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'constant',
                                                        target_height=40,
                                                        target_width=40)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=40,
                                                            min_y=40,
                                                            min_x=30)
        expected_patch = np.pad(expected_patch,
                                pad_width=((10, 10), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [10, 0, 30, 40]
        assert actual_roi == expected_roi

    def test_edge_constant_all_omage_no_pad(self, image):
        HEIGHT = 4
        WIDTH = 10
        roi = [0, 0, HEIGHT, WIDTH]
        actual_patch, transform = boxes_utils.cropPatch(
            image[0:HEIGHT, 0:WIDTH],
            roi,
            'constant',
            target_height=HEIGHT * 2,
            target_width=WIDTH)
        expected_patch = Test_CropPatch._make_gradient_mage(height=HEIGHT,
                                                            width=WIDTH,
                                                            min_y=0,
                                                            min_x=0)
        expected_patch = np.pad(expected_patch,
                                pad_width=((HEIGHT // 2, HEIGHT // 2), (0, 0),
                                           (0, 0)),
                                mode='constant')
        assert actual_patch.shape == expected_patch.shape
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [HEIGHT // 2, 0, HEIGHT * 3 // 2, WIDTH]
        assert actual_roi == expected_roi

    def test_edge_distort_Y(self, image):
        roi = [45, 40, 55, 60]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'distort',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=20,
                                                            min_y=45,
                                                            min_x=40,
                                                            max_y=55)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 0, 20, 20]
        assert actual_roi == expected_roi

    def test_edge_background_Y_no_stretch(self, image):
        roi = [45, 40, 55, 60]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=20,
                                                            min_y=40,
                                                            min_x=40)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        assert actual_roi == expected_roi

    def test_edge_background_X_no_stretch(self, image):
        roi = [40, 45, 60, 55]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=20,
                                                            min_y=40,
                                                            min_x=40)
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        assert actual_roi == expected_roi

    def test_edge_background_Y_below_0_no_stretch(self, image):
        roi = [0, 40, 10, 60]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=15,
                                                            width=20,
                                                            min_x=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((5, 0), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        assert actual_roi == expected_roi

    def test_edge_background_X_below_0_no_stretch(self, image):
        roi = [40, 0, 60, 10]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=15,
                                                            min_y=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 0), (5, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        assert actual_roi == expected_roi

    def test_edge_background_Y_above_top_no_stretch(self, image):
        roi = [90, 40, 100, 60]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=15,
                                                            width=20,
                                                            min_y=85,
                                                            min_x=40)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 5), (0, 0), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [5, 0, 15, 20]
        assert actual_roi == expected_roi

    def test_edge_background_X_above_top_no_stretch(self, image):
        roi = [40, 190, 60, 200]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=20,
                                                        target_width=20)
        expected_patch = Test_CropPatch._make_gradient_mage(height=20,
                                                            width=15,
                                                            min_y=40,
                                                            min_x=185)
        expected_patch = np.pad(expected_patch,
                                pad_width=((0, 0), (0, 5), (0, 0)),
                                mode='constant')
        np.testing.assert_array_equal(actual_patch, expected_patch)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [0, 5, 20, 15]
        assert actual_roi == expected_roi

    def test_edge_background_Y_below_0_stretch(self, image):
        roi = [0, 40, 10, 60]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=40,
                                                        target_width=40)
        expected_patch = Test_CropPatch._make_gradient_mage(height=30,
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
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [10, 0, 30, 40]
        assert actual_roi == expected_roi

    def test_edge_background_float_stretch(self, image):
        roi = [9.5, 9.5, 20.5, 30.5]
        # Compare patches.
        actual_patch, transform = boxes_utils.cropPatch(image,
                                                        roi,
                                                        'background',
                                                        target_height=40,
                                                        target_width=40)
        expected_patch = Test_CropPatch._make_gradient_mage(height=40,
                                                            width=40,
                                                            min_y=5,
                                                            min_x=10,
                                                            max_y=25,
                                                            max_x=30)
        norm = (actual_patch.mean() + expected_patch.mean()) / 2.
        actual_patch = actual_patch / norm
        expected_patch = expected_patch / norm
        np.testing.assert_array_almost_equal(actual_patch,
                                             expected_patch,
                                             decimal=0)
        # Compare roi.
        actual_roi = Test_CropPatch._transform_roi(transform, roi)
        expected_roi = [9, -1, 31, 41]
        assert actual_roi == expected_roi


class Test_getTransformBetweenRois:
    def test_identity(self):
        roi = [10, 20, 30, 40]
        np.testing.assert_array_equal(
            np.eye(3, 3, dtype=float),
            np.array(boxes_utils._getTransformBetweenRois(roi, roi)))

    def test_X2(self):
        roi1 = [10, 20, 30, 40]
        roi2 = [10, 20, 50, 60]
        np.testing.assert_array_equal(
            np.array([[2, 0, -10], [0, 2, -20], [0, 0, 1]]),
            np.array(boxes_utils._getTransformBetweenRois(roi1, roi2)))

    def test_invalid_roi1(self):
        roi1 = [10, 20, 0, 0]
        roi2 = [10, 20, 50, 60]
        with pytest.raises(ValueError):
            boxes_utils._getTransformBetweenRois(roi1, roi2)

    def test_invalid_roi2(self):
        roi1 = [10, 20, 50, 60]
        roi2 = [10, 20, 0, 0]
        with pytest.raises(ValueError):
            boxes_utils._getTransformBetweenRois(roi1, roi2)


class Test_ApplyTransformToRoi:
    def test_regular(self):
        roi = (10, 20, 30, 40)

        transform = np.array([[2, 0, 0], [0, 2, 0]])
        assert boxes_utils.applyTransformToRoi(transform,
                                               roi) == (20, 40, 60, 80)

        transform = np.array([[1, 0, 10], [0, 1, 20]])
        assert boxes_utils.applyTransformToRoi(transform,
                                               roi) == (20, 40, 40, 60)

        transform = np.array([[1, 1, 0], [1, 1, 0]])
        assert boxes_utils.applyTransformToRoi(transform,
                                               roi) == (30, 30, 70, 70)


class Test_ClipRoiToShape:
    def test_regular(self):
        shape = (100, 200, 3)
        # Normal.
        assert boxes_utils.clipRoiToShape((10, 20, 30, 40),
                                          shape) == (10, 20, 30, 40)
        # Out of boundaries.
        assert boxes_utils.clipRoiToShape((-10, 20, 30, 40),
                                          shape) == (0, 20, 30, 40)
        assert boxes_utils.clipRoiToShape((10, -20, 30, 40),
                                          shape) == (10, 0, 30, 40)
        assert boxes_utils.clipRoiToShape((10, 20, 300, 40),
                                          shape) == (10, 20, 100, 40)
        assert boxes_utils.clipRoiToShape((10, 20, 30, 400),
                                          shape) == (10, 20, 30, 200)
        # Float.
        assert boxes_utils.clipRoiToShape((10, 20, 30, 400.5),
                                          shape) == (10, 20, 30, 200)
        assert boxes_utils.clipRoiToShape((-10.5, 20, 30, 40),
                                          shape) == (0, 20, 30, 40)
