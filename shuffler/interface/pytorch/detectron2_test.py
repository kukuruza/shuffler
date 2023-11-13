import pytest
import os
import shutil
import tempfile
import numpy as np
import detectron2

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import detectron2 as shuffler_detectron2


class TestFilterKeys:
    def test_general(self):
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = shuffler_detectron2._filterKeys(['image'], sample)
        assert len(sample) == 1  # only "image" key is left.
        assert 'image' in sample


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
        shuffler_detectron2.register_object_dataset(
            "dataset1",
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR)
        dataset = detectron2.data.DatasetCatalog.get("dataset1")
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
        shuffler_detectron2.register_object_dataset(
            "dataset2",
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        dataset = detectron2.data.DatasetCatalog.get("dataset2")
        assert len(dataset) == 3  # 3 objects.
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert len(sample) == 2
        assert 'mask' in sample
        assert 'score' in sample

    def test_where_object(self, tmp_in_db_file):
        # All objects should be "cars".
        shuffler_detectron2.register_object_dataset(
            "dataset3",
            tmp_in_db_file,
            rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        dataset = detectron2.data.DatasetCatalog.get("dataset3")
        # 2 cars out of 3 objects in the dataset.
        assert len(dataset) == 2
