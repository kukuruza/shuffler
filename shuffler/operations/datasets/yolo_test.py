import os, os.path as op
import logging
import shutil
import progressbar
import argparse
import tempfile
import numpy as np
import nose

from shuffler.operations.datasets import yolo
from shuffler.utils import testing as testing_utils


class Test_exportYolo_carsDb(testing_utils.Test_carsDb):

    def setUp(self):
        super(Test_exportYolo_carsDb, self).setUp()
        self.temp_dir = op.join(tempfile.gettempdir(), 'Test_exportYolo')
        assert not op.exists(self.temp_dir)
        os.mkdir(self.temp_dir)

    def tearDown(self):
        super(Test_exportYolo_carsDb, self).tearDown()
        if op.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def helper(self, subset):
        # Check files.
        logging.debug('Contents of temp_dir: %s', os.listdir(self.temp_dir))
        images_subset_dir = op.join(self.temp_dir, 'images', subset)
        labels_subset_dir = op.join(self.temp_dir, 'labels', subset)
        logging.debug('Contents of images_subset_dir: %s',
                      os.listdir(images_subset_dir))
        logging.debug('Contents of labels_subset_dir: %s',
                      os.listdir(labels_subset_dir))
        self.assertTrue(op.exists(op.join(images_subset_dir, '000000.jpg')))
        self.assertTrue(op.exists(op.join(images_subset_dir, '000001.jpg')))
        self.assertTrue(op.exists(op.join(images_subset_dir, '000002.jpg')))
        self.assertTrue(op.exists(op.join(labels_subset_dir, '000000.txt')))
        self.assertTrue(op.exists(op.join(labels_subset_dir, '000001.txt')))
        # Empty images do not have labels file.
        self.assertFalse(op.exists(op.join(labels_subset_dir, '000002.txt')))

        # Actual labels.
        labels_actual0 = np.loadtxt(op.join(labels_subset_dir, '000000.txt'))
        labels_actual1 = np.loadtxt(op.join(labels_subset_dir, '000001.txt'))

        # Expected labels.
        test_data_dir = 'testdata/cars/yolo'
        expected_labels_subset_dir = op.join(test_data_dir, 'labels', subset)
        assert op.exists(op.join(expected_labels_subset_dir, '000000.txt'))
        assert op.exists(op.join(expected_labels_subset_dir, '000001.txt'))
        labels_expected0 = np.loadtxt(
            op.join(expected_labels_subset_dir, '000000.txt'))
        labels_expected1 = np.loadtxt(
            op.join(expected_labels_subset_dir, '000001.txt'))

        np.testing.assert_array_almost_equal(labels_actual0,
                                             labels_expected0,
                                             decimal=3)
        np.testing.assert_array_almost_equal(labels_actual1,
                                             labels_expected1,
                                             decimal=3)

    def test_carsOnly(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            yolo_dir=op.join(self.temp_dir),
            copy_images=True,
            symlink_images=False,
            subset='car',
            classes=['car'],
            as_polygons=False,
            dirtree_level_for_name=1,
            fix_invalid_image_names=False)
        yolo.exportYolo(c, args)
        self.helper('car')

    def test_carsAndBuses(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            yolo_dir=op.join(self.temp_dir),
            copy_images=True,
            symlink_images=False,
            subset='car_and_bus',
            classes=['car', 'bus'],
            as_polygons=False,
            dirtree_level_for_name=1,
            fix_invalid_image_names=False)
        yolo.exportYolo(c, args)
        self.helper('car_and_bus')

    def test_carsOnly_symlinkImages(self):
        ''' Only check if symlinks exist. '''
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            yolo_dir=op.join(self.temp_dir),
            copy_images=False,
            symlink_images=True,
            subset='car',
            classes=['car'],
            as_polygons=False,
            dirtree_level_for_name=1,
            fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        logging.debug('Contents of temp_dir: %s', os.listdir(self.temp_dir))
        images_dir = op.join(self.temp_dir, 'images/car')
        logging.debug('Contents of images_dir: %s', os.listdir(images_dir))
        # Check if exist.
        self.assertTrue(op.exists(op.join(images_dir, '000000.jpg')))
        self.assertTrue(op.exists(op.join(images_dir, '000001.jpg')))
        self.assertTrue(op.exists(op.join(images_dir, '000002.jpg')))
        # Check if a link.
        self.assertTrue(op.islink(op.join(images_dir, '000000.jpg')))
        self.assertTrue(op.islink(op.join(images_dir, '000001.jpg')))
        self.assertTrue(op.islink(op.join(images_dir, '000002.jpg')))

    def test_carsOnly_fullFilename(self):
        ''' Only check if image and label names are correct. '''
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            yolo_dir=op.join(self.temp_dir),
            copy_images=False,
            symlink_images=True,
            subset='car',
            classes=['car'],
            as_polygons=False,
            dirtree_level_for_name=2,
            fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        logging.debug('Contents of temp_dir: %s', os.listdir(self.temp_dir))
        images_dir = op.join(self.temp_dir, 'images/car')
        labels_dir = op.join(self.temp_dir, 'labels/car')
        logging.debug('Contents of images_dir: %s', os.listdir(images_dir))
        logging.debug('Contents of labels_dir: %s', os.listdir(labels_dir))
        self.assertTrue(op.exists(op.join(images_dir, 'images_000000.jpg')),
                        'Instead have: %s' % os.listdir(images_dir))
        self.assertTrue(op.exists(op.join(images_dir, 'images_000001.jpg')))
        self.assertTrue(op.exists(op.join(images_dir, 'images_000002.jpg')))
        self.assertTrue(op.exists(op.join(labels_dir, 'images_000000.txt')))
        self.assertTrue(op.exists(op.join(labels_dir, 'images_000001.txt')))

    def test_carsOnlyPolygons(self):
        ''' Eval only label files. They are written in the Polygon format. '''

        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            yolo_dir=op.join(self.temp_dir),
            copy_images=True,
            symlink_images=False,
            subset='car_as_polygons',
            classes=['car'],
            as_polygons=True,
            dirtree_level_for_name=1,
            fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        # Check files.
        logging.debug('Contents of temp_dir: %s', os.listdir(self.temp_dir))
        labels_subset_dir = op.join(self.temp_dir, 'labels/car_as_polygons')
        logging.debug('Contents of labels_subset_dir: %s',
                      os.listdir(labels_subset_dir))
        self.assertTrue(op.exists(op.join(labels_subset_dir, '000000.txt')))

        # Actual labels.
        labels_actual0 = np.loadtxt(op.join(labels_subset_dir, '000000.txt'))

        # Expected labels.
        test_data_dir = 'testdata/cars/yolo'
        expected_labels_subset_dir = op.join(test_data_dir,
                                             'labels/car_as_polygons')
        assert op.exists(op.join(expected_labels_subset_dir, '000000.txt'))
        labels_expected0 = np.loadtxt(
            op.join(expected_labels_subset_dir, '000000.txt'))

        np.testing.assert_array_almost_equal(labels_actual0,
                                             labels_expected0,
                                             decimal=3)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
