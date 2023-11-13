import pytest
import os
import shutil
import tempfile

from shuffler.utils import testing as testing_utils
from shuffler.interface.keras import generators


class TestListOfWhateverToWhateverOfLists:
    def test_empty(self):
        batch = generators._listOfWhateverToWhateverOfLists([])
        assert batch == []

    def test_list(self):
        # Regular.
        batch = generators._listOfWhateverToWhateverOfLists([['a', 1],
                                                             ['b', 2]])
        assert batch == [['a', 'b'], [1, 2]]
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([[], []])
        assert batch == []

    def test_tuple(self):
        # Regular.
        batch = generators._listOfWhateverToWhateverOfLists([('a', 1),
                                                             ('b', 2)])
        assert batch == (['a', 'b'], [1, 2])
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([(), ()])
        assert batch == ()

    def test_dict(self):
        # Regular.
        samples = [{'x': 'a', 'y': 1}, {'x': 'b', 'y': 2}]
        batch = generators._listOfWhateverToWhateverOfLists(samples)
        assert batch == {'x': ['a', 'b'], 'y': [1, 2]}
        # No keys.
        batch = generators._listOfWhateverToWhateverOfLists([{}, {}])
        assert batch == {}


class TestFilterKeys:
    @pytest.fixture()
    def sample(self):
        return {'x': 'a', 'y': 1}

    def test_none(self, sample):
        # Regular.
        filtered_sample = generators._filterKeys(None, sample)
        assert filtered_sample, sample
        # No keys and no filder does not raise an error.
        filtered_sample = generators._filterKeys(None, {})
        assert filtered_sample == {}

    def test_list(self, sample):
        # Regular.
        filtered_sample = generators._filterKeys(['x'], sample)
        assert filtered_sample == ['a']
        # No keys.
        with pytest.raises(KeyError):
            generators._filterKeys(['x'], {})

    def test_tuple(self, sample):
        # Regular.
        filtered_sample = generators._filterKeys(('x', ), sample)
        assert filtered_sample == ('a', )
        # No keys.
        with pytest.raises(KeyError):
            generators._filterKeys(('x', ), {})

    def test_dict(self, sample):
        # Regular.
        filtered_sample = generators._filterKeys({'x': 'X'}, sample)
        assert filtered_sample == {'X': 'a'}
        # No keys.
        with pytest.raises(KeyError):
            generators._filterKeys({'x': 'X'}, {})


class TestImageGenerator(testing_utils.CarsDb):
    @pytest.fixture()
    def tmp_in_db_file(self):
        ''' Copy the test database to a temp file to avoid damaging it. '''
        tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.CarsDb.CARS_DB_PATH, tmp_in_db_file)
        yield tmp_in_db_file
        if os.path.exists(tmp_in_db_file):
            os.remove(tmp_in_db_file)

    def test_general(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        assert isinstance(batch, dict)
        assert 'image' in batch
        assert isinstance(batch['image'], list)
        assert len(batch['image']) == batch_size
        assert 'mask' in batch
        assert 'objects' in batch
        assert 'imagefile' in batch
        assert 'name' in batch
        assert 'score' in batch
        assert isinstance(batch['objects'], list)
        assert '_index_' in batch  # index of an item in Dataset.

        # sample['objects'] has a list of dicts. Check each dict.
        assert len(batch['objects']) == batch_size
        objects_in_sample = batch['objects'][0]
        assert isinstance(objects_in_sample, list)
        object_in_sample = objects_in_sample[0]
        assert isinstance(object_in_sample, dict)
        assert 'x1' in object_in_sample
        assert 'y1' in object_in_sample
        assert 'width' in object_in_sample
        assert 'height' in object_in_sample
        assert 'name' in object_in_sample
        assert 'score' in object_in_sample

    def test_used_keys_list(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'],
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        assert isinstance(batch, list)
        assert len(batch) == 2

    def test_used_keys_dict(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ImageGenerator(
            testing_utils.CarsDb.CARS_DB_PATH,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys={
                'mask': 'mask_',
                'score': 'score_'
            },
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches (with 2 and 1 images).
        batch = generator[0]
        assert isinstance(batch, dict)
        assert len(batch) == 2
        assert 'mask_' in batch
        assert 'score_' in batch

    def test_where_image(self, tmp_in_db_file):
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_image='imagefile == "images/000000.jpg"')
        assert len(generator) == 1  # 1 image.

    def test_where_object(self, tmp_in_db_file):
        # All objects should be "cars".
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        batch = generator[0]
        assert len(batch['objects']) == 1  # 1 car in the 1st image.

        # All objects should be "bus".
        generator = generators.ImageGenerator(
            testing_utils.CarsDb.CARS_DB_PATH,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "bus"')
        batch = generator[0]
        assert len(batch['objects'][0]) == 0  # No buses in the 1st image.


class TestObjectGenerator(testing_utils.CarsDb):
    @pytest.fixture()
    def tmp_in_db_file(self):
        ''' Copy the test database to a temp file to avoid damaging it. '''
        tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.CarsDb.CARS_DB_PATH, tmp_in_db_file)
        yield tmp_in_db_file
        if os.path.exists(tmp_in_db_file):
            os.remove(tmp_in_db_file)

    def test_general(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ObjectGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            mode='w',
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches, with 1 and 2 objects.
        batch = generator[0]
        assert isinstance(batch, dict)
        assert 'image' in batch
        assert 'mask' in batch
        assert 'objectid' in batch
        assert 'imagefile' in batch
        assert 'name' in batch
        assert 'score' in batch
        assert '_index_' in batch  # index of an item in Dataset.

        # Add a record.
        generator.addRecord(batch['objectid'][0], "result", "0.5")
        # Test that the record was added as expected.
        generator.c.execute(
            'SELECT value FROM properties WHERE objectid=? AND key="result"',
            (batch['objectid'][0], ))
        assert generator.c.fetchall() == [("0.5", )]

    def test_used_keys_list(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'],
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches (with 2 and 1 objects)
        batch = generator[0]
        assert isinstance(batch, list)
        assert len(batch) == 2

    def test_used_keys_dict(self, tmp_in_db_file):
        batch_size = 2
        generator = generators.ImageGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys={
                'mask': 'mask_',
                'score': 'score_'
            },
            batch_size=batch_size)
        assert len(generator) == 2  # 2 batches (with 2 and 1 objects)
        batch = generator[0]
        assert isinstance(batch, dict)
        assert len(batch) == 2
        assert 'mask_' in batch
        assert 'score_' in batch

    def test_where_object(self, tmp_in_db_file):
        # All objects should be "cars".
        generator = generators.ObjectGenerator(
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        # 2 cars out of 3 objects in the generator.
        assert len(generator) == 2  # Each batch is one car.
