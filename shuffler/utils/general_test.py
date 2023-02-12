import os.path as op
import numpy as np
import imageio
import cv2
import shutil
import unittest
import tempfile
import progressbar
import nose

from shuffler.utils import general as general_utils


class Test_drawFilledRoi(unittest.TestCase):
    RED = (255, 0, 0)

    def test_regular_fullfill(self):
        ''' fill_opacity=1 must fill the area completely. '''
        image = imageio.imread('imageio:chelsea.png')
        expected_image = image.copy()
        expected_image[50:150, 100:200] = self.RED

        general_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                     self.RED,
                                     fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_float_roi(self):
        ''' Test that it does not break for non-integer ROI. '''
        image = imageio.imread('imageio:chelsea.png')
        general_utils._drawFilledRoi(image, (50.3, 100.7, 150.1, 200),
                                     self.RED,
                                     fill_opacity=1)

    def test_regular_nofill(self):
        ''' fill_opacity=0 must have no effect on the image. '''
        image = imageio.imread('imageio:chelsea.png')
        expected_image = image.copy()
        # Do NOT change the image, because fill_opacity=0.

        general_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                     self.RED,
                                     fill_opacity=0)
        # np.testing.assert_array_equal(image, expected_image)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_grayscale_fullfill(self):
        ''' Grayscale images should be processed correctly. '''
        image = imageio.imread('imageio:camera.png')
        assert len(image.shape) == 2, 'camera.png was expected to be grayscale'
        expected_image = image.copy()
        expected_image[50:150, 100:200] = 255

        general_utils._drawFilledRoi(image, (50, 100, 150, 200),
                                     255,
                                     fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary1_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[0:150, 200:451] = self.RED

        general_utils._drawFilledRoi(image, (-100, 200, 150, 600),
                                     self.RED,
                                     fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary2_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[150:300, 0:200] = self.RED

        general_utils._drawFilledRoi(image, (150, -100, 400, 200),
                                     self.RED,
                                     fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)


class Test_drawFilledPolygon(unittest.TestCase):
    RED = (255, 0, 0)

    def test_regular_fullfill(self):
        ''' fill_opacity=1 must fill the area completely. '''
        image = imageio.imread('imageio:chelsea.png')
        expected_image = image.copy()
        expected_image[50:150, 100:300] = self.RED

        general_utils._drawFilledPolygon(image, [(50, 100), (50, 300),
                                                 (150, 300), (150, 100)],
                                         self.RED,
                                         fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_float_polygon(self):
        ''' Test that it does not break for non-integer polygon. '''
        image = imageio.imread('imageio:chelsea.png')
        general_utils._drawFilledPolygon(image, [(50.1, 100.2), (50.3, 300.4),
                                                 (150.5, 300.6), (150, 100)],
                                         self.RED,
                                         fill_opacity=1)

    def test_invalid_polygon(self):
        ''' Bad polygons must be quietly ignored. '''
        image = imageio.imread('imageio:chelsea.png')
        expected_image = image.copy()

        general_utils._drawFilledPolygon(image, [(50, 100), (50, 200)],
                                         self.RED,
                                         fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_regular_nofill(self):
        ''' fill_opacity=0 must have no effect on the image. '''
        image = imageio.imread('imageio:chelsea.png')
        expected_image = image.copy()
        # Do NOT change the image, because fill_opacity=0.

        general_utils._drawFilledPolygon(image, [(50, 100), (50, 200),
                                                 (150, 200), (150, 100)],
                                         self.RED,
                                         fill_opacity=0)
        # np.testing.assert_array_equal(image, expected_image)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_grayscale_fullfill(self):
        ''' Grayscale images should be processed correctly. '''
        image = imageio.imread('imageio:camera.png')
        assert len(image.shape) == 2, 'camera.png was expected to be grayscale'
        expected_image = image.copy()
        expected_image[50:150, 100:200] = 255

        general_utils._drawFilledPolygon(image, [(50, 100), (50, 200),
                                                 (150, 200), (150, 100)],
                                         255,
                                         fill_opacity=1)
        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary1_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[0:150, 200:451] = self.RED

        general_utils._drawFilledPolygon(image, [(-100, 200), (-100, 600),
                                                 (150, 600), (150, 200)],
                                         self.RED,
                                         fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)

    def test_off_boundary2_fullfill(self):
        ''' ROI out of image boundary must be processed correctly. '''
        image = imageio.imread('imageio:chelsea.png')  # 300 x 451 x 3
        expected_image = image.copy()
        expected_image[150:350, 0:200] = self.RED

        general_utils._drawFilledPolygon(image, [(150, -100), (150, 200),
                                                 (500, 200), (500, -100)],
                                         self.RED,
                                         fill_opacity=1)

        # The boundary may differ a tiny bit.
        np.testing.assert_almost_equal(image, expected_image, 0.0001)


class Test_CopyWithBackup(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        with open(op.join(self.work_dir, 'from.txt'), 'w') as f:
            f.write('from')

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_failsIfFileNotExists(self):
        from_path = op.join(self.work_dir, 'not_exists.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        with self.assertRaises(FileNotFoundError):
            general_utils.copyWithBackup(from_path, to_path)

    def test_copiesWithoutNeedForBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        general_utils.copyWithBackup(from_path, to_path)
        self.assertTrue(op.exists(from_path))
        self.assertTrue(op.exists(to_path))

    def test_copiesWithBackup(self):
        from_path = op.join(self.work_dir, 'from.txt')
        to_path = op.join(self.work_dir, 'to.txt')
        backup_path = op.join(self.work_dir, 'to.backup.txt')
        # Make an existing file.
        with open(to_path, 'w') as f:
            f.write('old')
        general_utils.copyWithBackup(from_path, to_path)
        # Make sure the contents are overwritten.
        self.assertTrue(op.exists(to_path))
        with open(to_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'old')

    def test_copiestoItself(self):
        from_path = op.join(self.work_dir, 'from.txt')
        backup_path = op.join(self.work_dir, 'from.backup.txt')
        general_utils.copyWithBackup(from_path, from_path)
        # Make sure the contents are the same.
        self.assertTrue(op.exists(from_path))
        with open(from_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')
        # Make sure the old file was backed up.
        self.assertTrue(op.exists(backup_path))
        with open(backup_path) as f:
            s = f.readline()
            self.assertEqual(s, 'from')


class Test_getIntersectingObjects(unittest.TestCase):
    def test_empty(self):
        pairs_to_merge = general_utils.getIntersectingObjects([], [], 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_firstEmpty(self):
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects([], objects2,
                                                              0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_identical(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [(1, 2)])

    def test_identical_sameId(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(1, 'image', 10, 10, 30, 30, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(objects1,
                                                              objects2,
                                                              0.5,
                                                              same_id_ok=False)
        self.assertEqual(pairs_to_merge, [])

    def test_nonIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 40, 40, 'name2', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.5)
        self.assertEqual(pairs_to_merge, [])

    def test_twoIntersecting(self):
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.)]
        objects2 = [(2, 'image', 20, 20, 30, 30, 'name2', 1.),
                    (3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 40, 50, 60, 70, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(pairs_to_merge, [(1, 3)])

    def test_twoAndTwoIntersecting(self):
        # #1.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 20, 20, 30, 30, 'name3', 1.),
                    (4, 'image', 10, 10, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 4), (2, 3)]))
        # #2.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 20, 20, 30, 30, 'name4', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 4)]))
        # #3.
        objects1 = [(1, 'image', 10, 10, 30, 30, 'name1', 1.),
                    (2, 'image', 20, 20, 30, 30, 'name2', 1.)]
        objects2 = [(3, 'image', 10, 10, 30, 30, 'name3', 1.),
                    (4, 'image', 0, 0, 30, 30, 'name4', 1.),
                    (5, 'image', 20, 20, 30, 30, 'name5', 1.)]
        pairs_to_merge = general_utils.getIntersectingObjects(
            objects1, objects2, 0.1)
        self.assertEqual(set(pairs_to_merge), set([(1, 3), (2, 5)]))


class Test_makeExportedImageName(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_regular(self):
        tgt_path = general_utils.makeExportedImageName(self.work_dir,
                                                       'src_dir/filename')
        self.assertEqual(tgt_path, op.join(self.work_dir, 'filename'))

        # Make it exist.
        with open(tgt_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            general_utils.makeExportedImageName(self.work_dir,
                                                'src_dir/filename')

    def test_dirtreeLevelForName_eq2(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=2)
        self.assertEqual(tgt_path, 'tgt_dir/fancy_filename')

    def test_dirtreeLevelForName_eq3(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=3)
        self.assertEqual(tgt_path, 'tgt_dir/my_fancy_filename')

    def test_dirtreeLevelForName_eqALot(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'my/fancy/filename', dirtree_level_for_name=42)
        self.assertEqual(tgt_path, 'tgt_dir/my_fancy_filename')

    def test_fixInvalidImageNames(self):
        tgt_path = general_utils.makeExportedImageName(
            'tgt_dir', 'src_dir/file(?)name', fix_invalid_image_names=True)
        self.assertEqual(tgt_path, 'tgt_dir/file___name')


class Test_getMatchPolygons(unittest.TestCase):
    def test_empty(self):
        pairs = general_utils.getMatchPolygons([], [], 1.)
        self.assertEqual(pairs, [])

    def test_firstEmpty(self):
        objectid = 1
        polygons2 = [(1, objectid, 10, 30, 'name1')]
        pairs = general_utils.getMatchPolygons([], polygons2, 1.)
        self.assertEqual(pairs, [])

    def test_identical(self):
        ''' Identical points are not matched if when ignoring names. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., True)
        self.assertEqual(pairs, [(1, 2)])

    def test_identicalAndNameMatter(self):
        ''' Identical points are not matched if names differ. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1')]
        polygons2 = [(2, objectid, 10, 30, 'name2')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(pairs, [])

    def test_someMatchingPointsAndNameIgnored(self):
        ''' Some points are matched, names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 100, 100, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., True)
        self.assertEqual(pairs, [(2, 3)])

    def test_ambiguousPointsInPolygon1(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        # If there are matches of repeated points, then smth is wrong here.
        with self.assertRaises(ValueError):
            general_utils.getMatchPolygons(polygons1, polygons2, 1., True)

    def test_ambiguousPointsInPolygon2(self):
        ''' Expect an error if two points can be matched. Names are ignored. '''
        objectid = 1
        polygons1 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        polygons2 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        # If there are matches of repeated points, then smth is wrong here.
        with self.assertRaises(ValueError):
            general_utils.getMatchPolygons(polygons1, polygons2, 1., True)

    def test_haveRepeatedPointsAndNameMatter(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, 'name1'),
                     (2, objectid, 10, 30, 'name2')]
        polygons2 = [(3, objectid, 10, 30, 'name2'),
                     (4, objectid, 200, 200, 'name3')]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(pairs, [(2, 3)])

    def test_haveNameIsNull(self):
        ''' Only the point with matching name is matched out of two points. '''
        objectid = 1
        polygons1 = [(1, objectid, 10, 30, None),
                     (2, objectid, 20, 40, 'name1'),
                     (3, objectid, 30, 50, 'name2')]
        polygons2 = [(4, objectid, 30, 50, 'name2'),
                     (5, objectid, 20, 40, 'name1'),
                     (6, objectid, 10, 30, None)]
        pairs = general_utils.getMatchPolygons(polygons1, polygons2, 1., False)
        self.assertEqual(set(pairs), set([(1, 6), (2, 5), (3, 4)]))


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
