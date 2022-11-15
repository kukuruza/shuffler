import progressbar
import unittest
import numpy as np

from shuffler.interface import utils


class TestApplyTransform(unittest.TestCase):
    def test_callable(self):
        transform_group = lambda x: x['image']
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = utils.applyTransformGroup(transform_group, sample)
        self.assertTrue(isinstance(sample, (np.ndarray, np.generic)))

    def test_list(self):
        transform_group = [lambda x: x['image'], lambda x: x['objectid']]
        sample = {'image': np.zeros(3), 'objectid': 1, 'name': 'car'}
        sample = utils.applyTransformGroup(transform_group, sample)
        self.assertTrue(isinstance(sample, list))
        self.assertEqual(len(sample), 2)  # only "image" and 'objectid' left.
        self.assertTrue(isinstance(sample[0], (np.ndarray, np.generic)))
        self.assertTrue(isinstance(sample[1], int))

    def test_dict(self):
        transform_group = {
            'image': lambda x: x[:, :, 0],
            'name': lambda _: 'hi',
            'dummy': lambda x: x,
        }
        sample = {'image': np.zeros((10, 10, 3)), 'objectid': 1, 'name': 'car'}
        sample = utils.applyTransformGroup(transform_group, sample)
        self.assertEqual(len(sample['image'].shape), 2)  # Grayscale image.
        self.assertEqual(sample['name'], 'hi')  # All names replaced to "hi".
        self.assertIsNone(sample['dummy'])  # Returns None for missing keys.


class TestCheckTransformGroup(unittest.TestCase):
    def test_callable(self):
        transform_group = lambda x: x['image']
        sample = {'image': np.zeros(3), 'objectid': 1}
        utils.checkTransformGroup(transform_group)

    def test_list(self):
        transform_group = [lambda x: x['image'], lambda x: x['objectid']]
        sample = {'image': np.zeros(3), 'objectid': 1, 'name': 'car'}
        utils.checkTransformGroup(transform_group)

    def test_dict(self):
        transform_group = {
            'image': lambda x: x[:, :, 0],
            'name': lambda _: 'hi'
        }
        sample = {'image': np.zeros((10, 10, 3)), 'objectid': 1, 'name': 'car'}
        utils.checkTransformGroup(transform_group)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    unittest.main()
