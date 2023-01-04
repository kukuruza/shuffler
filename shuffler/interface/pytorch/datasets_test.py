import os
import progressbar
import shutil
import tempfile
import unittest
import numpy as np

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import datasets


class TestHelperFunctions(unittest.TestCase):

    def test_filterKeys(self):
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = datasets._filterKeys(['image'], sample)
        self.assertEqual(len(sample), 1)  # only "image" key is left.
        self.assertTrue('image' in sample)


class TestImageDataset(testing_utils.Test_carsDb):

    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.Test_carsDb.CARS_DB_PATH,
                    self.tmp_in_db_file)

    def tearDown(self):
        if os.path.exists(self.tmp_in_db_file):
            os.remove(self.tmp_in_db_file)

    def test_general(self):
        dataset = datasets.ImageDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR)
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

    def test_used_keys(self):
        dataset = datasets.ImageDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        self.assertEqual(len(dataset), 3)  # 3 images.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertEqual(len(sample), 2)
        self.assertTrue('mask' in sample)
        self.assertTrue('score' in sample)

    def test_where_image(self):
        dataset = datasets.ImageDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_image='imagefile == "images/000000.jpg"')
        self.assertEqual(len(dataset), 1)  # 1 image.

    def test_where_object(self):
        # All objects should be "cars".
        dataset = datasets.ImageDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        sample = dataset[0]
        self.assertEqual(len(sample['objects']), 1)  # 1 car in the 1st image.

        # All objects should be "bus".
        dataset = datasets.ImageDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "bus"')
        sample = dataset[0]
        self.assertEqual(len(sample['objects']),
                         0)  # No buses in the 1st image.


class TestObjectDataset(testing_utils.Test_carsDb):

    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.Test_carsDb.CARS_DB_PATH,
                    self.tmp_in_db_file)

    def tearDown(self):
        if os.path.exists(self.tmp_in_db_file):
            os.remove(self.tmp_in_db_file)

    def test_general(self):
        dataset = datasets.ObjectDataset(
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
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

    def test_general_preload(self):
        dataset = datasets.ObjectDataset(
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            preload_samples=True)
        self.assertEqual(len(dataset), 3)  # 3 objects.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertTrue('image' in sample)
        self.assertTrue('mask' in sample)
        self.assertTrue('objectid' in sample)
        self.assertTrue('imagefile' in sample)
        self.assertTrue('name' in sample)
        self.assertTrue('score' in sample)

    def test_used_keys(self):
        dataset = datasets.ObjectDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        self.assertEqual(len(dataset), 3)  # 3 objects.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertEqual(len(sample), 2)
        self.assertTrue('mask' in sample)
        self.assertTrue('score' in sample)

    def test_where_object(self):
        # All objects should be "cars".
        dataset = datasets.ObjectDataset(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        # 2 cars out of 3 objects in the dataset.
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    unittest.main()
