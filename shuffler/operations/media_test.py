import pytest
import unittest.mock
import argparse
import numpy as np

from shuffler.operations import media
from shuffler.operations import modify
from shuffler.utils import testing as testing_utils


class TestCropObjects_EmptyDb(testing_utils.EmptyDb):
    def assert_empty(self, c):
        c.execute('SELECT COUNT(1) FROM images')
        assert c.fetchone()[0] == 0
        c.execute('SELECT COUNT(1) FROM objects')
        assert c.fetchone()[0] == 0
        c.execute('SELECT COUNT(1) FROM polygons')
        assert c.fetchone()[0] == 0
        c.execute('SELECT COUNT(1) FROM matches')
        assert c.fetchone()[0] == 0
        c.execute('SELECT COUNT(1) FROM properties')
        assert c.fetchone()[0] == 0

    def test_keep_all_objects(self, c):
        args = argparse.Namespace(rootdir='.',
                                  media='mock',
                                  image_path='a',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_empty(c)

    def test_general(self, c):
        args = argparse.Namespace(rootdir='.',
                                  media='mock',
                                  image_path='a',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_empty(c)


class TestCropObjects_CarsDb(testing_utils.CarsDb):
    def test_general(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path='mock_mask_media',
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        self.assert_polygons_count_by_object(c, expected=[0, 5, 0])
        self.assert_objects_count_by_match(c, expected=[2])
        # +6 for 6 new properties "original_objectid","crop","kx","ky","bx","by"
        self.assert_properties_count_by_object(c,
                                               expected=[3 + 6, 3 + 6, 1 + 6])
        # Check that maskfiles were written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        assert c.fetchone()[0] == 3

    def test_mask_path_not_provided(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        # Check that maskfiles were NOT written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        assert c.fetchone()[0] == 0

    def test_mask_does_not_exist(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path='mock_mask_media',
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        c.execute('UPDATE images SET maskfile=NULL')
        media.cropObjects(c, args)
        # Check that maskfiles were NOT written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        assert c.fetchone()[0] == 0

    @unittest.mock.patch('shuffler.operations.media.backend_media.MediaWriter')
    def test_video_not_allowed_if_edges_original(self, _, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='video',
                                  image_path='mock_video.avi',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        with pytest.raises(Exception):
            media.cropObjects(c, args)

    def test_only_buses(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='objects.name="bus"',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_images_count(c, expected=1)
        self.assert_objects_count_by_imagefile(c, expected=[1])
        self.assert_polygons_count_by_object(c, expected=[0])
        self.assert_objects_count_by_match(c, expected=[])
        # +6 for 6 new properties "original_objectid","crop","kx","ky","bx","by"
        self.assert_properties_count_by_object(c, expected=[1 + 6])

    def test_keep_all_other_objects(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='TRUE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 2, 2])
        self.assert_polygons_count_by_object(c, expected=[0, 0, 5, 5, 0])
        self.assert_objects_count_by_match(c, expected=[3])

    def test_keep_other_buses(self, c):
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='name="bus"',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 2, 1])
        self.assert_polygons_count_by_object(c, expected=[0, 5, 0, 0])
        self.assert_objects_count_by_match(c, expected=[2])

    @unittest.mock.patch('shuffler.operations.media.backend_media.MediaWriter')
    def test_namehint(self, mock_imwriter, c):
        ''' Test 'namehint'. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='pictures',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='objectid IN (2, 3)',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=2)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1])
        # imwrite must be called 2 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        assert len(call_args_list) == 2
        assert call_args_list[0][1] == {'namehint': '000000002'}
        assert call_args_list[1][1] == {'namehint': '000000003'}

    @unittest.mock.patch('shuffler.operations.media.backend_media.MediaWriter')
    def test_namehint_splitIntoFoldersByObjectName(self, mock_imwriter, c):
        ''' Test 'namehint' when split_into_folders_by_object_name is on. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='pictures',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=True,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        # imwrite must be called 3 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        assert len(call_args_list) == 3
        assert call_args_list[0][1] == {'namehint': 'car/000000001'}
        assert call_args_list[1][1] == {'namehint': 'car/000000002'}
        assert call_args_list[2][1] == {'namehint': 'bus/000000003'}

    @unittest.mock.patch('shuffler.operations.media.backend_media.MediaWriter')
    def test_namehint_addObjectNameToFilename(self, mock_imwriter, c):
        ''' Test 'namehint' when add_object_name_to_filename is on. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='pictures',
                                  image_path='mock_media',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=True)
        media.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        # imwrite must be called 3 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        assert len(call_args_list) == 3
        assert call_args_list[0][1] == {'namehint': 'car-000000001'}
        assert call_args_list[1][1] == {'namehint': 'car-000000002'}
        assert call_args_list[2][1] == {'namehint': 'bus-000000003'}


class TestCropObjects_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",0,40,20,40,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,40,20)')
        yield c

    @unittest.mock.patch('shuffler.operations.media.boxes_utils.cropPatch')
    @unittest.mock.patch.object(media.backend_media.MediaReader, 'imread')
    def test_xy(self, mocked_imread, mocked_crop_patch, c):
        mocked_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        transform = np.array([[2., 0., -5.], [0., 0.5, 5.], [0., 0., 1.]])
        mocked_crop_patch.return_value = (np.zeros((100, 100, 3)), transform)
        args = argparse.Namespace(rootdir='.',
                                  media='mock',
                                  image_path='a',
                                  mask_path=None,
                                  where_object='TRUE',
                                  where_other_objects='FALSE',
                                  target_width=None,
                                  target_height=None,
                                  edges='original',
                                  overwrite=False,
                                  split_into_folders_by_object_name=False,
                                  add_object_name_to_filename=False)
        media.cropObjects(c, args)
        c.execute('SELECT x1,y1,width,height FROM objects')
        x1, y1, width, height = c.fetchone()
        assert (x1, y1, width, height) == (25, 35, 20, 40)
        c.execute('SELECT x,y FROM polygons')
        x, y = c.fetchone()
        assert (x, y) == (25, 35)
        # Check that all temporary tables are cleaned up.
        c.execute('SELECT name FROM sqlite_master WHERE type="table"')
        table_names = c.fetchall()
        assert len(table_names) == 5, table_names
        # Check the object transform.
        c.execute('SELECT value FROM properties WHERE key="kx"')
        kx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="ky"')
        ky = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="bx"')
        bx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="by"')
        by = c.fetchall()
        assert (len(kx), len(ky), len(bx), len(by)) == (1, 1, 1, 1)
        transform_recorded = np.array([[float(ky[0][0]), 0.,
                                        float(by[0][0])],
                                       [0.,
                                        float(kx[0][0]),
                                        float(bx[0][0])], [0., 0., 1.]])
        np.testing.assert_array_equal(transform, transform_recorded)


class TestTileObjects_CarsDb(testing_utils.CarsDb):
    @staticmethod
    def _test_consistency(c, num_cells_Y, num_cells_X, split_by_name,
                          image_icon):
        '''
        Test that running tileObjects and undoing that transformation gives
        identify.
        '''
        # Copy all tables ("matches" are not supported).
        c.execute("CREATE TABLE objects_ref AS SELECT * FROM objects")
        c.execute("CREATE TABLE properties_ref AS SELECT * FROM properties")
        c.execute("CREATE TABLE polygons_ref AS SELECT * FROM polygons")
        # Run tiling.
        args = argparse.Namespace(rootdir=testing_utils.CarsDb.CARS_DB_ROOTDIR,
                                  media='mock',
                                  image_path='mock_media',
                                  mask_path='mock_mask_media',
                                  where_object='TRUE',
                                  num_cells_Y=num_cells_Y,
                                  num_cells_X=num_cells_X,
                                  inter_cell_gap=10,
                                  cell_width=50,
                                  cell_height=50,
                                  edges='constant',
                                  overwrite=False,
                                  split_by_name=split_by_name,
                                  image_icon=image_icon)
        media.tileObjects(c, args)
        # Run reverting.
        modify.revertObjectTransforms(c, argparse.Namespace())

        # Compare the result with the reference for "objects".
        c.execute("SELECT objectid FROM objects_ref")
        objects_ref = c.fetchall()
        c.execute(
            "SELECT objects.objectid FROM objects INNER JOIN objects_ref "
            "ON objects.objectid = objects_ref.objectid WHERE "
            "objects.imagefile IS objects_ref.imagefile AND "
            "ABS(objects.width - objects_ref.width) < 2 AND "
            "ABS(objects.height - objects_ref.height) < 2 AND "
            "ABS(objects.x1 - objects_ref.x1) < 2 AND "
            "ABS(objects.y1 - objects_ref.y1) < 2 AND "
            "objects.name IS objects_ref.name AND "
            "objects.score IS objects_ref.score")
        same_objects = c.fetchall()
        assert set(objects_ref) == set(same_objects)

        # Compare the result with the reference for "polygons".
        c.execute("SELECT objectid FROM polygons_ref")
        polygons_ref = c.fetchall()
        c.execute(
            "SELECT polygons.objectid FROM polygons INNER JOIN polygons_ref "
            "ON polygons.objectid = polygons_ref.objectid WHERE "
            "polygons.x IS polygons_ref.x AND "
            "polygons.y IS polygons_ref.y")
        same_polygons = c.fetchall()
        assert set(polygons_ref) == set(same_polygons)

        # Compare the result with the reference for "properties".
        c.execute("SELECT objectid FROM properties_ref")
        properties_ref = c.fetchall()
        c.execute(
            "SELECT properties.objectid FROM properties INNER JOIN properties_ref "
            "ON properties.objectid = properties_ref.objectid WHERE "
            "properties.key IS properties_ref.key AND "
            "properties.value IS properties_ref.value")
        same_properties = c.fetchall()
        assert set(properties_ref) == set(same_properties)

    def test_consistency1(self, c):
        TestTileObjects_CarsDb._test_consistency(c, 2, 1, False, False)

    def test_consistency2(self, c):
        TestTileObjects_CarsDb._test_consistency(c, 1, 1, False, False)

    def test_consistency3(self, c):
        TestTileObjects_CarsDb._test_consistency(c, 2, 1, False, False)

    def test_consistency4(self, c):
        TestTileObjects_CarsDb._test_consistency(c, 1, 1, True, False)

    def test_consistency5(self, c):
        TestTileObjects_CarsDb._test_consistency(c, 1, 1, False, True)
