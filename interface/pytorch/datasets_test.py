import os
import progressbar
import shutil
import tempfile
import unittest
import numpy as np

from lib.utils import testUtils
from interface.pytorch import datasets


class TestHelperFunctions(testUtils.Test_carsDb):
    def test_used_keys(self):
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = datasets._filter_keys(['image'], sample)
        self.assertEqual(len(sample), 1)  # only "image" key is left.
        self.assertTrue('image' in sample)

    def test_apply_transform_callable(self):
        transform_group = lambda x: x['image']
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = datasets._apply_transform(transform_group, sample)
        self.assertTrue(isinstance(sample, (np.ndarray, np.generic)))

    def test_apply_transform_list(self):
        transform_group = [lambda x: x['image'], lambda x: x['objectid']]
        sample = {'image': np.zeros(3), 'objectid': 1, 'name': 'car'}
        sample = datasets._apply_transform(transform_group, sample)
        self.assertTrue(isinstance(sample, list))
        self.assertEqual(len(sample), 2)  # only "image" and 'objectid' left.
        self.assertTrue(isinstance(sample[0], (np.ndarray, np.generic)))
        self.assertTrue(isinstance(sample[1], int))

    def test_apply_transform_dict(self):
        transform_group = {
            'image': lambda x: x[:, :, 0],
            'name': lambda _: 'hi'
        }
        sample = {'image': np.zeros((10, 10, 3)), 'objectid': 1, 'name': 'car'}
        sample = datasets._apply_transform(transform_group, sample)
        self.assertEqual(len(sample['image'].shape), 2)  # Grayscale image.
        self.assertEqual(sample['name'], 'hi')  # All names replaced to "hi".


class TestImageDataset(testUtils.Test_carsDb):
    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testUtils.Test_carsDb.CARS_DB_PATH, self.tmp_in_db_file)

    def tearDown(self):
        os.remove(self.tmp_in_db_file)

    def test_general(self):
        dataset = datasets.ImageDataset(
            testUtils.Test_carsDb.CARS_DB_PATH,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR)
        self.assertEqual(len(dataset), 3)  # 3 images.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertTrue('image' in sample)
        self.assertTrue('mask' in sample)
        self.assertTrue('objects' in sample)
        self.assertTrue('imagefile' in sample)
        self.assertTrue('name' in sample)
        self.assertTrue('score' in sample)
        self.assertTrue(isinstance(sample['objects'], list))

        # sample['objects'] has a list of dicts. Check each dict.
        self.assertEqual(len(sample['objects']), 1)
        object_sample = sample['objects'][0]
        self.assertTrue(isinstance(object_sample, dict))
        self.assertTrue('x1' in object_sample)
        self.assertTrue('y1' in object_sample)
        self.assertTrue('width' in object_sample)
        self.assertTrue('height' in object_sample)
        self.assertTrue('name' in object_sample)
        self.assertTrue('score' in object_sample)

    def test_where_image(self):
        dataset = datasets.ImageDataset(
            testUtils.Test_carsDb.CARS_DB_PATH,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            where_image='imagefile == "images/000000.jpg"')
        self.assertEqual(len(dataset), 1)  # 1 image.

    def test_where_object(self):
        # All objects should be "cars".
        dataset = datasets.ImageDataset(
            testUtils.Test_carsDb.CARS_DB_PATH,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        sample = dataset[0]
        self.assertEqual(len(sample['objects']), 1)  # 1 car in the 1st image.

        # All objects should be "bus".
        dataset = datasets.ImageDataset(
            testUtils.Test_carsDb.CARS_DB_PATH,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "bus"')
        sample = dataset[0]
        self.assertEqual(len(sample['objects']),
                         0)  # No buses in the 1st image.


class TestObjectDataset(testUtils.Test_carsDb):
    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testUtils.Test_carsDb.CARS_DB_PATH, self.tmp_in_db_file)

    def tearDown(self):
        os.remove(self.tmp_in_db_file)

    def test_general(self):
        dataset = datasets.ObjectDataset(
            self.tmp_in_db_file,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            mode='w')
        self.assertEqual(len(dataset), 3)  # 3 objects.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertTrue('image' in sample)
        self.assertTrue('mask' in sample)
        self.assertTrue('objectid' in sample)
        self.assertTrue('imagefile' in sample)
        self.assertTrue('name' in sample)
        self.assertTrue('score' in sample)

        # Add a record.
        dataset.addRecord(sample['objectid'], "result", "0.5")
        # Test that the record was added as expected.
        dataset.c.execute(
            'SELECT value FROM properties WHERE objectid=? AND key="result"',
            (sample['objectid'], ))
        self.assertEqual(dataset.c.fetchall(), [("0.5", )])

    def test_where_object(self):
        # All objects should be "cars".
        dataset = datasets.ObjectDataset(
            testUtils.Test_carsDb.CARS_DB_PATH,
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        # 2 cars out of 3 objects in the dataset.
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    unittest.main()
