import os, os.path as op
import logging
import sqlite3
import shutil
import progressbar
import unittest
import argparse
import pprint
import tempfile
import mock
import numpy as np
import nose

from lib.backend import backendDb
from lib.subcommands import dbMedia
from lib.utils import testUtils


class Test_cropObjects_emptyDb(testUtils.Test_emptyDb):
    def assertEmpty(self, c):
        c.execute('SELECT COUNT(1) FROM images')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(1) FROM objects')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(1) FROM polygons')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(1) FROM matches')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(1) FROM properties')
        self.assertEqual(c.fetchone()[0], 0)

    def test_keep_all_objects(self):
        c = self.conn.cursor()
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
        dbMedia.cropObjects(c, args)
        self.assertEmpty(c)

    def test_general(self):
        c = self.conn.cursor()
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
        dbMedia.cropObjects(c, args)
        self.assertEmpty(c)


class Test_cropObjects_carsDb(testUtils.Test_carsDb):
    def test_general(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        self.assert_polygons_count_by_object(c, expected=[0, 5, 0])
        self.assert_objects_count_by_match(c, expected=[2])
        # +1 is for the new property "crop" and "kx","ky","bx","by".
        self.assert_properties_count_by_object(c,
                                               expected=[3 + 5, 3 + 5, 1 + 5])
        # Check that maskfiles were written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        self.assertEqual(c.fetchone()[0], 3)

    def test_mask_path_not_provided(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        # Check that maskfiles were NOT written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        self.assertEqual(c.fetchone()[0], 0)

    def test_mask_does_not_exist(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        # Check that maskfiles were NOT written
        c.execute(
            'SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
        self.assertEqual(c.fetchone()[0], 0)

    @mock.patch('lib.subcommands.dbMedia.backendMedia.MediaWriter')
    def test_video_not_allowed_if_edges_original(self, mock_imwriter):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        with self.assertRaises(Exception):
            dbMedia.cropObjects(c, args)

    def test_only_buses(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        self.assert_images_count(c, expected=1)
        self.assert_objects_count_by_imagefile(c, expected=[1])
        self.assert_polygons_count_by_object(c, expected=[0])
        self.assert_objects_count_by_match(c, expected=[])
        # +5 is for the new property "crop" and "kx, ky, bx, by".
        self.assert_properties_count_by_object(c, expected=[5])

    def test_keep_all_other_objects(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 2, 2])
        self.assert_polygons_count_by_object(c, expected=[0, 0, 5, 5, 0])
        self.assert_objects_count_by_match(c, expected=[3])

    def test_keep_other_buses(self):
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 2, 1])
        self.assert_polygons_count_by_object(c, expected=[0, 5, 0, 0])
        self.assert_objects_count_by_match(c, expected=[2])

    @mock.patch('lib.subcommands.dbMedia.backendMedia.MediaWriter')
    def test_namehint(self, mock_imwriter):
        ''' Test 'namehint'. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=2)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1])
        # imwrite must be called 2 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        self.assertEqual(len(call_args_list), 2)
        self.assertEqual(call_args_list[0][1], {'namehint': '000000002'})
        self.assertEqual(call_args_list[1][1], {'namehint': '000000003'})

    @mock.patch('lib.subcommands.dbMedia.backendMedia.MediaWriter')
    def test_namehint_splitIntoFoldersByObjectName(self, mock_imwriter):
        ''' Test 'namehint' when split_into_folders_by_object_name is on. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        # imwrite must be called 3 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        self.assertEqual(len(call_args_list), 3)
        self.assertEqual(call_args_list[0][1], {'namehint': 'car/000000001'})
        self.assertEqual(call_args_list[1][1], {'namehint': 'car/000000002'})
        self.assertEqual(call_args_list[2][1], {'namehint': 'bus/000000003'})

    @mock.patch('lib.subcommands.dbMedia.backendMedia.MediaWriter')
    def test_namehint_addObjectNameToFilename(self, mock_imwriter):
        ''' Test 'namehint' when add_object_name_to_filename is on. '''
        mock_imwriter.return_value.imwrite.side_effect = ['foo', 'bar', 'baz']
        c = self.conn.cursor()
        args = argparse.Namespace(
            rootdir=testUtils.Test_carsDb.CARS_DB_ROOTDIR,
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
        dbMedia.cropObjects(c, args)
        # Check that the crops are correct.
        self.assert_images_count(c, expected=3)
        self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
        # imwrite must be called 3 times.
        call_args_list = mock_imwriter.return_value.imwrite.call_args_list
        self.assertEqual(len(call_args_list), 3)
        self.assertEqual(call_args_list[0][1], {'namehint': 'car 000000001'})
        self.assertEqual(call_args_list[1][1], {'namehint': 'car 000000002'})
        self.assertEqual(call_args_list[2][1], {'namehint': 'bus 000000003'})


class Test_cropObjects_SyntheticDb(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backendDb.createDb(self.conn)
        c = self.conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",0,40,20,40,20)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,40,20)')

    @mock.patch('lib.subcommands.dbMedia.utilBoxes.cropPatch')
    @mock.patch.object(dbMedia.backendMedia.MediaReader, 'imread')
    def test_xy(self, mocked_imread, mocked_crop_patch):
        mocked_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        transform = np.array([[2., 0., -5.], [0., 0.5, 5.], [0., 0., 1.]])
        mocked_crop_patch.return_value = (np.zeros((100, 100, 3)), transform)
        c = self.conn.cursor()
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
        dbMedia.cropObjects(c, args)
        c.execute('SELECT x1,y1,width,height FROM objects')
        x1, y1, width, height = c.fetchone()
        self.assertEqual((x1, y1, width, height), (25, 35, 20, 40))
        c.execute('SELECT x,y FROM polygons')
        x, y = c.fetchone()
        self.assertEqual((x, y), (25, 35))
        # Check that all temporary tables are cleaned up.
        c.execute('SELECT name FROM sqlite_master WHERE type="table"')
        table_names = c.fetchall()
        self.assertEqual(len(table_names), 5, table_names)
        # Check the object transform.
        c.execute('SELECT value FROM properties WHERE key="kx"')
        kx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="ky"')
        ky = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="bx"')
        bx = c.fetchall()
        c.execute('SELECT value FROM properties WHERE key="by"')
        by = c.fetchall()
        self.assertEqual((len(kx), len(ky), len(bx), len(by)), (1, 1, 1, 1))
        transform_recorded = np.array([[float(ky[0][0]), 0.,
                                        float(by[0][0])],
                                       [0.,
                                        float(kx[0][0]),
                                        float(bx[0][0])], [0., 0., 1.]])

        np.testing.assert_array_equal(transform, transform_recorded)


if __name__ == '__main__':
    progressbar.streams.wrap_stdout()
    nose.runmodule()
