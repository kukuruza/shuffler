'''
A demo for running Keras inference on a database, and recording some results.
'''

import os
import progressbar
import shutil
import tempfile
import unittest

from shuffler.utils import testing as testing_utils
from shuffler.interface.keras import generators


class TestListOfWhateverToWhateverOfLists(unittest.TestCase):

    def test_empty(self):
        batch = generators._listOfWhateverToWhateverOfLists([])
        self.assertEqual(batch, [])

    def test_list(self):
        # Regular.
        batch = generators._listOfWhateverToWhateverOfLists([['a', 1],
                                                             ['b', 2]])
        self.assertEqual(batch, [['a', 'b'], [1, 2]])
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([[], []])
        self.assertEqual(batch, [])

    def test_tuple(self):
        # Regular.
        batch = generators._listOfWhateverToWhateverOfLists([('a', 1),
                                                             ('b', 2)])
        self.assertEqual(batch, (['a', 'b'], [1, 2]))
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([(), ()])
        self.assertEqual(batch, ())

    def test_dict(self):
        # Regular.
        samples = [{'x': 'a', 'y': 1}, {'x': 'b', 'y': 2}]
        batch = generators._listOfWhateverToWhateverOfLists(samples)
        self.assertEqual(batch, {'x': ['a', 'b'], 'y': [1, 2]})
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([{}, {}])
        self.assertEqual(batch, {})


class TestFilterKeys(unittest.TestCase):

    def setUp(self):
        self.sample = {'x': 'a', 'y': 1}

    def test_none(self):
        # Regular.
        sample = generators._filterKeys(None, self.sample)
        self.assertEqual(sample, self.sample)
        # No keys and no filder does not raise an error.
        sample = generators._filterKeys(None, {})
        self.assertEqual(sample, {})

    def test_list(self):
        # Regular.
        sample = generators._filterKeys(['x'], self.sample)
        self.assertEqual(sample, ['a'])
        # No keys.
        with self.assertRaises(KeyError):
            generators._filterKeys(['x'], {})

    def test_tuple(self):
        # Regular.
        sample = generators._filterKeys(('x', ), self.sample)
        self.assertEqual(sample, ('a', ))
        # No keys.
        with self.assertRaises(KeyError):
            generators._filterKeys(('x', ), {})

    def test_dict(self):
        # Regular.
        sample = generators._filterKeys({'x': 'X'}, self.sample)
        self.assertEqual(sample, {'X': 'a'})
        # No keys.
        with self.assertRaises(KeyError):
            generators._filterKeys({'x': 'X'}, {})


class TestImageGenerator(testing_utils.Test_carsDb):

    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.Test_carsDb.CARS_DB_PATH,
                    self.tmp_in_db_file)

    def tearDown(self):
        if os.path.exists(self.tmp_in_db_file):
            os.remove(self.tmp_in_db_file)

    def test_general(self):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        self.assertTrue(isinstance(batch, dict))
        self.assertTrue('image' in batch)
        self.assertTrue(isinstance(batch['image'], list))
        self.assertEqual(len(batch['image']), batch_size)
        self.assertTrue('mask' in batch)
        self.assertTrue('objects' in batch)
        self.assertTrue('imagefile' in batch)
        self.assertTrue('name' in batch)
        self.assertTrue('score' in batch)
        self.assertTrue(isinstance(batch['objects'], list))

        # sample['objects'] has a list of dicts. Check each dict.
        self.assertEqual(len(batch['objects']), batch_size)
        objects_in_sample = batch['objects'][0]
        self.assertTrue(isinstance(objects_in_sample, list))
        object_in_sample = objects_in_sample[0]
        self.assertTrue(isinstance(object_in_sample, dict))
        self.assertTrue('x1' in object_in_sample)
        self.assertTrue('y1' in object_in_sample)
        self.assertTrue('width' in object_in_sample)
        self.assertTrue('height' in object_in_sample)
        self.assertTrue('name' in object_in_sample)
        self.assertTrue('score' in object_in_sample)

    def test_used_keys_list(self):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'],
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        self.assertTrue(isinstance(batch, list))
        self.assertEqual(len(batch), 2)

    def test_used_keys_dict(self):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys={
                'mask': 'mask_',
                'score': 'score_'
            },
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        self.assertTrue(isinstance(batch, dict))
        self.assertEqual(len(batch), 2)
        self.assertTrue('mask_' in batch)
        self.assertTrue('score_' in batch)

    def test_where_image(self):
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_image='imagefile == "images/000000.jpg"')
        self.assertEqual(len(generator), 1)  # 1 image.

    def test_where_object(self):
        # All objects should be "cars".
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        batch = generator[0]
        self.assertEqual(len(batch['objects']), 1)  # 1 car in the 1st image.

        # All objects should be "bus".
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "bus"')
        batch = generator[0]
        self.assertEqual(len(batch['objects'][0]),
                         0)  # No buses in the 1st image.


class TestObjectGenerator(testing_utils.Test_carsDb):

    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.Test_carsDb.CARS_DB_PATH,
                    self.tmp_in_db_file)

    def tearDown(self):
        if os.path.exists(self.tmp_in_db_file):
            os.remove(self.tmp_in_db_file)

    def test_general(self):
        batch_size = 2
        generator = generators.ObjectGenerator(
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            mode='w',
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches, with 1 and 2 objects.
        batch = generator[0]
        self.assertTrue(isinstance(batch, dict))
        self.assertTrue('image' in batch)
        self.assertTrue('mask' in batch)
        self.assertTrue('objectid' in batch)
        self.assertTrue('imagefile' in batch)
        self.assertTrue('name' in batch)
        self.assertTrue('score' in batch)

        # Add a record.
        generator.addRecord(batch['objectid'][0], "result", "0.5")
        # Test that the record was added as expected.
        generator.c.execute(
            'SELECT value FROM properties WHERE objectid=? AND key="result"',
            (batch['objectid'][0], ))
        self.assertEqual(generator.c.fetchall(), [("0.5", )])

    def test_used_keys_list(self):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'],
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches (with 2 and 1 objects)
        batch = generator[0]
        self.assertTrue(isinstance(batch, list))
        self.assertEqual(len(batch), 2)

    def test_used_keys_dict(self):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys={
                'mask': 'mask_',
                'score': 'score_'
            },
            batch_size=batch_size)
        self.assertEqual(len(generator), 2)  # 2 batches (with 2 and 1 objects)
        batch = generator[0]
        self.assertTrue(isinstance(batch, dict))
        self.assertEqual(len(batch), 2)
        self.assertTrue('mask_' in batch)
        self.assertTrue('score_' in batch)

    def test_where_object(self):
        # All objects should be "cars".
        generator = generators.ObjectGenerator(
            testing_utils.Test_carsDb.CARS_DB_PATH,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        # 2 cars out of 3 objects in the generator.
        self.assertEqual(len(generator), 2)  # Each batch is one car.


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    unittest.main()
