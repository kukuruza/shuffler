import os
import progressbar
import shutil
import tempfile
import unittest
import numpy as np
import detectron2

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import detectron2 as shuffler_detectron2


class TestHelperFunctions(unittest.TestCase):

    def test_filterKeys(self):
        sample = {'image': np.zeros(3), 'objectid': 1}
        sample = shuffler_detectron2._filterKeys(['image'], sample)
        self.assertEqual(len(sample), 1)  # only "image" key is left.
        self.assertTrue('image' in sample)


class TestObjectDataset(testing_utils.Test_carsDb):

    def setUp(self):
        self.tmp_in_db_file = tempfile.NamedTemporaryFile().name
        shutil.copy(testing_utils.Test_carsDb.CARS_DB_PATH,
                    self.tmp_in_db_file)

    def tearDown(self):
        if os.path.exists(self.tmp_in_db_file):
            os.remove(self.tmp_in_db_file)

    def test_general(self):
        shuffler_detectron2.register_object_dataset(
            "dataset1",
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR)
        dataset = detectron2.data.DatasetCatalog.get("dataset1")
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
        shuffler_detectron2.register_object_dataset(
            "dataset2",
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            used_keys=['mask', 'score'])
        dataset = detectron2.data.DatasetCatalog.get("dataset2")
        self.assertEqual(len(dataset), 3)  # 3 objects.
        sample = dataset[0]
        self.assertTrue(isinstance(sample, dict))
        self.assertEqual(len(sample), 2)
        self.assertTrue('mask' in sample)
        self.assertTrue('score' in sample)

    def test_where_object(self):
        # All objects should be "cars".
        shuffler_detectron2.register_object_dataset(
            "dataset3",
            self.tmp_in_db_file,
            rootdir=testing_utils.Test_carsDb.CARS_DB_ROOTDIR,
            where_object='name == "car"')
        dataset = detectron2.data.DatasetCatalog.get("dataset3")
        # 2 cars out of 3 objects in the dataset.
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    unittest.main()
