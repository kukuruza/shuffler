import pytest
import os
import shutil
import tempfile
import numpy as np

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import datasets


class TestFilterKeys:
    def test_general(self):
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = datasets._filterKeys(['image'], sample)
        assert len(sample) == 1  # only "image" key is left.
        assert 'image' in sample


class TestImageDataset(testing_utils.CarsDb):
    @pytest.fixture()
    def tmp_in_db_file(self):
        ''' Copy the test database to a temp file to avoid damaging it. '''
        tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.CarsDb.CARS_DB_PATH, tmp_in_db_file)
        yield tmp_in_db_file
        if os.path.exists(tmp_in_db_file):
            os.remove(tmp_in_db_file)

    def test_general(self, tmp_in_db_file):
        dataset = datasets.ImageDataset(
            tmp_in_db_file, rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR)
        assert len(dataset) == 3  # 3 images.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'mask' in sample
        assert 'objects' in sample
        assert 'imagefile' in sample
        assert 'name' in sample
        assert 'score' in sample
        assert isinstance(sample['objects'], list)
        assert '_index_' in sample  # index of an item in Dataset.

        # sample['objects'] has a list of dicts. Check each dict.
        assert len(sample['objects']) == 1
        object_sample = sample['objects'][0]
        assert isinstance(object_sample, dict)
        assert 'x1' in object_sample
        assert 'y1' in object_sample
        assert 'width' in object_sample
        assert 'height' in object_sample
        assert 'name' in object_sample
        assert 'score' in object_sample

    def test_used_keys(self, tmp_in_db_file):
        dataset = datasets.ImageDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        assert len(dataset) == 3  # 3 images.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert len(sample) == 2
        assert 'mask' in sample
        assert 'score' in sample

    def test_where_image(self, tmp_in_db_file):
        dataset = datasets.ImageDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_image='imagefile == "images/000000.jpg"')
        assert len(dataset) == 1  # 1 image.

    def test_where_object(self, tmp_in_db_file):
        # All objects should be "cars".
        dataset = datasets.ImageDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        sample = dataset[0]
        assert len(sample['objects']) == 1  # 1 car in the 1st image.

        # All objects should be "bus".
        dataset = datasets.ImageDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "bus"')
        sample = dataset[0]
        assert len(sample['objects']) == 0  # No buses in the 1st image.


class TestObjectDataset(testing_utils.CarsDb):
    @pytest.fixture()
    def tmp_in_db_file(self):
        ''' Copy the test database to a temp file to avoid damaging it. '''
        tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.CarsDb.CARS_DB_PATH, tmp_in_db_file)
        yield tmp_in_db_file
        if os.path.exists(tmp_in_db_file):
            os.remove(tmp_in_db_file)

    def test_general(self, tmp_in_db_file):
        dataset = datasets.ObjectDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            mode='w')
        assert len(dataset) == 3  # 3 objects.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'mask' in sample
        assert 'objectid' in sample
        assert 'imagefile' in sample
        assert 'name' in sample
        assert 'score' in sample
        assert '_index_' in sample  # index of an item in Dataset.

        # Add a record.
        dataset.addRecord(sample['objectid'], "result", "0.5")
        # Test that the record was added as expected.
        dataset.c.execute(
            'SELECT value FROM properties WHERE objectid=? AND key="result"',
            (sample['objectid'], ))
        assert dataset.c.fetchall() == [("0.5", )]

    def test_general_preload(self, tmp_in_db_file):
        dataset = datasets.ObjectDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            preload_samples=True)
        assert len(dataset) == 3  # 3 objects.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'mask' in sample
        assert 'objectid' in sample
        assert 'imagefile' in sample
        assert 'name' in sample
        assert 'score' in sample

    def test_used_keys(self, tmp_in_db_file):
        dataset = datasets.ObjectDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        assert len(dataset) == 3  # 3 objects.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert len(sample) == 2
        assert 'mask' in sample
        assert 'score' in sample

    def test_where_object(self, tmp_in_db_file):
        # All objects should be "cars".
        dataset = datasets.ObjectDataset(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        # 2 cars out of 3 objects in the dataset.
        assert len(dataset) == 2
