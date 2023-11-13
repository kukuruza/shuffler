import pytest
import os, os.path as op
import logging
import shutil
import argparse
import tempfile
import numpy as np

from shuffler.operations.datasets import yolo
from shuffler.utils import testing as testing_utils


class Test_exportYolo_CarsDb(testing_utils.CarsDb):
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def helper(self, work_dir, subset):
        # Check files.
        logging.debug('Contents of temp_dir: %s', os.listdir(work_dir))
        images_subset_dir = op.join(work_dir, 'images', subset)
        labels_subset_dir = op.join(work_dir, 'labels', subset)
        logging.debug('Contents of images_subset_dir: %s',
                      os.listdir(images_subset_dir))
        logging.debug('Contents of labels_subset_dir: %s',
                      os.listdir(labels_subset_dir))
        assert op.exists(op.join(images_subset_dir, '000000.jpg'))
        assert op.exists(op.join(images_subset_dir, '000001.jpg'))
        assert op.exists(op.join(images_subset_dir, '000002.jpg'))
        assert op.exists(op.join(labels_subset_dir, '000000.txt'))
        assert op.exists(op.join(labels_subset_dir, '000001.txt'))
        # Empty images do not have labels file.
        assert not op.exists(op.join(labels_subset_dir, '000002.txt'))

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

    def test_cars_only(self, c, work_dir):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  yolo_dir=op.join(work_dir),
                                  copy_images=True,
                                  symlink_images=False,
                                  subset='car',
                                  classes=['car'],
                                  as_polygons=False,
                                  dirtree_level_for_name=1,
                                  fix_invalid_image_names=False)
        yolo.exportYolo(c, args)
        self.helper(work_dir, 'car')

    def test_carsAndBuses(self, c, work_dir):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  yolo_dir=op.join(work_dir),
                                  copy_images=True,
                                  symlink_images=False,
                                  subset='car_and_bus',
                                  classes=['car', 'bus'],
                                  as_polygons=False,
                                  dirtree_level_for_name=1,
                                  fix_invalid_image_names=False)
        yolo.exportYolo(c, args)
        self.helper(work_dir, 'car_and_bus')

    def test_cars_only__symlink_images(self, c, work_dir):
        ''' Only check if symlinks exist. '''
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  yolo_dir=op.join(work_dir),
                                  copy_images=False,
                                  symlink_images=True,
                                  subset='car',
                                  classes=['car'],
                                  as_polygons=False,
                                  dirtree_level_for_name=1,
                                  fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        logging.debug('Contents of temp_dir: %s', os.listdir(work_dir))
        images_dir = op.join(work_dir, 'images/car')
        logging.debug('Contents of images_dir: %s', os.listdir(images_dir))
        # Check if exist.
        assert op.exists(op.join(images_dir, '000000.jpg'))
        assert op.exists(op.join(images_dir, '000001.jpg'))
        assert op.exists(op.join(images_dir, '000002.jpg'))
        # Check if a link.
        assert op.islink(op.join(images_dir, '000000.jpg'))
        assert op.islink(op.join(images_dir, '000001.jpg'))
        assert op.islink(op.join(images_dir, '000002.jpg'))

    def test_cars_only__full_filename(self, c, work_dir):
        ''' Only check if image and label names are correct. '''
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  yolo_dir=op.join(work_dir),
                                  copy_images=False,
                                  symlink_images=True,
                                  subset='car',
                                  classes=['car'],
                                  as_polygons=False,
                                  dirtree_level_for_name=2,
                                  fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        logging.debug('Contents of temp_dir: %s', os.listdir(work_dir))
        images_dir = op.join(work_dir, 'images/car')
        labels_dir = op.join(work_dir, 'labels/car')
        logging.debug('Contents of images_dir: %s', os.listdir(images_dir))
        logging.debug('Contents of labels_dir: %s', os.listdir(labels_dir))
        assert op.exists(op.join(
            images_dir,
            'images_000000.jpg')), 'Instead have: %s' % os.listdir(images_dir)
        assert op.exists(op.join(images_dir, 'images_000001.jpg'))
        assert op.exists(op.join(images_dir, 'images_000002.jpg'))
        assert op.exists(op.join(labels_dir, 'images_000000.txt'))
        assert op.exists(op.join(labels_dir, 'images_000001.txt'))

    def test_cars_only__polygons(self, c, work_dir):
        ''' Eval only label files. They are written in the Polygon format. '''

        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  yolo_dir=op.join(work_dir),
                                  copy_images=True,
                                  symlink_images=False,
                                  subset='car_as_polygons',
                                  classes=['car'],
                                  as_polygons=True,
                                  dirtree_level_for_name=1,
                                  fix_invalid_image_names=False)
        yolo.exportYolo(c, args)

        # Check files.
        logging.debug('Contents of temp_dir: %s', os.listdir(work_dir))
        labels_subset_dir = op.join(work_dir, 'labels/car_as_polygons')
        logging.debug('Contents of labels_subset_dir: %s',
                      os.listdir(labels_subset_dir))
        assert op.exists(op.join(labels_subset_dir, '000000.txt'))

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
