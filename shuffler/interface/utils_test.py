import pytest
import numpy as np

from shuffler.interface import utils


class TestApplyTransform:
    def test_callable(self):
        transform_group = lambda x: x['image']
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = utils.applyTransformGroup(transform_group, sample)
        assert isinstance(sample, (np.ndarray, np.generic))

    def test_list(self):
        transform_group = [lambda x: x['image'], lambda x: x['objectid']]
        sample = {'image': np.zeros(3), 'objectid': 1, 'name': 'car'}
        sample = utils.applyTransformGroup(transform_group, sample)
        assert isinstance(sample, list)
        assert len(sample) == 2  # only "image" and 'objectid' left.
        assert isinstance(sample[0], (np.ndarray, np.generic))
        assert isinstance(sample[1], int)

    def test_dict(self):
        transform_group = {
            'image': lambda x: x[:, :, 0],
            'name': lambda _: 'hi',
            'dummy': lambda x: x,
        }
        sample = {'image': np.zeros((10, 10, 3)), 'objectid': 1, 'name': 'car'}
        sample = utils.applyTransformGroup(transform_group, sample)
        assert len(sample['image'].shape) == 2  # Grayscale image.
        assert sample['name'] == 'hi'  # All names replaced to "hi".
        assert sample['dummy'] is None  # Returns None for missing keys.


class TestCheckTransformGroup:
    def test_callable(self):
        transform_group = lambda x: x['image']
        utils.checkTransformGroup(transform_group)

    def test_list(self):
        transform_group = [lambda x: x['image'], lambda x: x['objectid']]
        utils.checkTransformGroup(transform_group)

    def test_dict(self):
        transform_group = {
            'image': lambda x: x[:, :, 0],
            'name': lambda _: 'hi'
        }
        utils.checkTransformGroup(transform_group)
