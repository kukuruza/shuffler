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

from lib import backendDb
from lib import dbMedia


CARS_DB_PATH = 'lib/testdata/cars/micro1_v4.db'
CARS_DB_ROOTDIR = 'lib/testdata'


class Test_DB(unittest.TestCase):
  ''' Implements useful functions to compare the result db with the expected one. '''

  def summarize_db(self, c):
    summary = []
    summary.append('--- DB summary start ---')
    c.execute('SELECT imagefile FROM images')
    summary.append("images:")
    summary.append(pprint.pformat(c.fetchall()))
    c.execute('SELECT imagefile, objectid FROM objects')
    summary.append("objects (imagefile, objectid):")
    summary.append(pprint.pformat(c.fetchall()))
    c.execute('SELECT objectid, key, value FROM properties')
    summary.append("properties (objectid, key, value):")
    summary.append(pprint.pformat(c.fetchall()))
    c.execute('SELECT objectid, COUNT(1) FROM polygons GROUP BY objectid')
    summary.append("polygons (objectid, COUNT):")
    summary.append(pprint.pformat(c.fetchall()))
    c.execute('SELECT objectid, match FROM matches')
    summary.append("matches (objectid, match):")
    summary.append(pprint.pformat(c.fetchall()))
    summary.append('--- DB summary end -----')
    return '\n' + '\n'.join(summary)
  
  def verify_that_expected_is_int(self, expected):
    if not isinstance(expected, int):
      raise TypeError('"expected" should be int, not %s' % type(expected))

  def verify_that_expected_is_a_list_of_ints(self, expected):
    if not isinstance(expected, list):
      raise TypeError('"expected" should be list, not %s' % type(expected))
    if len(expected) > 0 and not isinstance(expected[0], int):
      raise TypeError('each element of "expected" should be int, not %s' % type(expected[0]))

  def assert_images_count(self, c, expected):
    '''
    Check the number of images.
    Args:
      c:          Cursor.
      expected:   Int, count of all images.
    '''
    self.verify_that_expected_is_int(expected)
    c.execute('SELECT COUNT(1) FROM images')
    self.assertEqual(c.fetchone()[0], expected, self.summarize_db(c))

  def assert_objects_count_by_imagefile(self, c, expected):
    '''
    Check the number of objects grouped by imagefile.
    Args:
      c:          Cursor.
      expected:   A list of ints, each element is a number of objects in one imagefile.
    '''
    self.verify_that_expected_is_a_list_of_ints(expected)
    c.execute(
      'SELECT COUNT(i.imagefile) FROM images i LEFT OUTER JOIN objects o '
      'ON i.imagefile = o.imagefile GROUP BY i.imagefile')
    actual = c.fetchall()
    expected = [(x,) for x in expected]
    self.assertEqual(sorted(actual), sorted(expected), self.summarize_db(c))

  def assert_polygons_count_by_object(self, c, expected):
    '''
    Check the number of polygon points grouped by objectid.
    Args:
      c:          Cursor.
      expected:   A list of ints, each element is a number of polygon point for one objectid.
                  The order of elements is not important.
    '''
    self.verify_that_expected_is_a_list_of_ints(expected)
    c.execute(
      'SELECT COUNT(p.objectid) FROM objects o LEFT OUTER JOIN polygons p '
      'ON p.objectid = o.objectid GROUP BY o.objectid')
    actual = c.fetchall()
    expected = [(x,) for x in expected]
    self.assertEqual(sorted(actual), sorted(expected), self.summarize_db(c))

  def assert_objects_count_by_match(self, c, expected):
    '''
    Check the number of objects grouped by match.
    Args:
      c:          Cursor.
      expected:   A list of ints, each element is a number of objects for one match.
                  The order of elements is not important.
    '''
    self.verify_that_expected_is_a_list_of_ints(expected)
    c.execute('SELECT COUNT(1) FROM matches GROUP BY match')
    actual = c.fetchall()
    expected = [(x,) for x in expected]
    self.assertEqual(sorted(actual), sorted(expected), self.summarize_db(c))


  def assert_properties_count_by_object(self, c, expected):
    '''
    Check the number of properties grouped by objectid.
    Args:
      c:          Cursor.
      expected:   A list of ints, each element is a number of properties for one objectid.
                  The order of elements is not important.
    '''
    self.verify_that_expected_is_a_list_of_ints(expected)
    c.execute(
      'SELECT COUNT(p.objectid) FROM objects o LEFT OUTER JOIN properties p '
      'ON o.objectid = p.objectid GROUP BY o.objectid')
    actual = c.fetchall()
    expected = [(x,) for x in expected]
    self.assertEqual(sorted(actual), sorted(expected), self.summarize_db(c))


class Test_emptyDb (Test_DB):

  def setUp (self):
    self.conn = sqlite3.connect(':memory:')
    backendDb.createDb(self.conn)

  def tearDown (self):
    self.conn.close()


class Test_carsDb (Test_DB):
  '''
  carsDb: image/000000:
            objectids:
              1. name: car, properties: yaw, pitch, color
          image/000001:
            objectids:  
              2. name: car, 5 polygons, properties: yaw, pitch, color
              3. name: bus, properties: yaw
          image/000003:
            objectsids: NA
          matches:
            match 1: objectids 1 and 2
  '''
  def setUp (self):
    self.temp_db_path = tempfile.NamedTemporaryFile().name
    shutil.copyfile(CARS_DB_PATH, self.temp_db_path)
    self.conn = sqlite3.connect(self.temp_db_path)

  def tearDown (self):
    self.conn.close()
    os.remove(self.temp_db_path)


class Test_cropObjects_emptyDb (Test_emptyDb):

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
    args = argparse.Namespace(
      rootdir='.', media='mock', image_path='a', mask_path=None,
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assertEmpty(c)

  def test_general(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir='.', media='mock', image_path='a', mask_path=None, 
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assertEmpty(c)


class Test_cropObjects_carsDb (Test_carsDb):

  def test_general(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path='mock_mask_media', 
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assert_images_count(c, expected=3)
    self.assert_objects_count_by_imagefile(c, expected=[1, 1, 1])
    self.assert_polygons_count_by_object(c, expected=[0, 5, 0])
    self.assert_objects_count_by_match(c, expected=[2])
    # +1 is for the new property "crop": "true" of the cropped object.
    self.assert_properties_count_by_object(c, expected=[3+1, 3+1, 1+1])
    # Check that maskfiles were written
    c.execute('SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
    self.assertEqual(c.fetchone()[0], 3)

  def test_mask_path_not_provided(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path=None, 
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    # Check that maskfiles were NOT written
    c.execute('SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
    self.assertEqual(c.fetchone()[0], 0)

  def test_mask_does_not_exist(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path='mock_mask_media', 
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    c.execute('UPDATE images SET maskfile=NULL')
    dbMedia.cropObjects(c, args)
    # Check that maskfiles were NOT written
    c.execute('SELECT COUNT(maskfile) FROM images WHERE maskfile IS NOT NULL')
    self.assertEqual(c.fetchone()[0], 0)

  def test_video_not_allowed_if_edges_original(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='video', image_path='mock_video.avi', mask_path=None, 
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    with self.assertRaises(Exception):
      dbMedia.cropObjects(c, args)

  def test_only_buses(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path=None,
      where_object='objects.name="bus"', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assert_images_count(c, expected=1)
    self.assert_objects_count_by_imagefile(c, expected=[1])
    self.assert_polygons_count_by_object(c, expected=[0])
    self.assert_objects_count_by_match(c, expected=[])
    # +1 is for the new property "crop": "true" of the cropped object.
    self.assert_properties_count_by_object(c, expected=[1])

  def test_keep_all_other_objects(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path=None,
      where_object='TRUE', where_other_objects='TRUE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assert_images_count(c, expected=3)
    self.assert_objects_count_by_imagefile(c, expected=[1, 2, 2])
    self.assert_polygons_count_by_object(c, expected=[0, 0, 5, 5, 0])
    self.assert_objects_count_by_match(c, expected=[3])
    # +1 is for the new property "crop": "true" of the cropped object.
    self.assert_properties_count_by_object(c, expected=[3+1, 3+1, 1, 1+1, 3])

  def test_keep_other_buses(self):
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir=CARS_DB_ROOTDIR, media='mock', image_path='mock_media', mask_path=None,
      where_object='TRUE', where_other_objects='name="bus"',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    self.assert_images_count(c, expected=3)
    self.assert_objects_count_by_imagefile(c, expected=[1, 2, 1])
    self.assert_polygons_count_by_object(c, expected=[0, 5, 0, 0])
    self.assert_objects_count_by_match(c, expected=[2])
    # +1 is for the new property "crop": "true" of the cropped object.
    self.assert_properties_count_by_object(c, expected=[3+1, 3+1, 1, 1+1])


class Test_cropObjects_SyntheticDb (unittest.TestCase):

  def setUp (self):
    self.conn = sqlite3.connect(':memory:')
    backendDb.createDb(self.conn)
    c = self.conn.cursor()
    c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
    c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
              'VALUES ("image0",0,40,20,40,20)')
    c.execute('INSERT INTO polygons(objectid,x,y) VALUES (0,40,20)')

  @mock.patch('lib.dbMedia.utilBoxes.cropPatch')
  @mock.patch.object(dbMedia.backendMedia.MediaReader, 'imread')
  def test_xy(self, mocked_imread, mocked_crop_patch):
    mocked_imread.return_value = np.zeros((100,100,3), dtype=np.uint8)
    transform = np.array([[2., 0., -5.], [0., 0.5, 5.]])
    mocked_crop_patch.return_value=(np.zeros((100,100,3)), transform)
    c = self.conn.cursor()
    args = argparse.Namespace(
      rootdir='.', media='mock', image_path='a', mask_path=None,
      where_object='TRUE', where_other_objects='FALSE',
      target_width=None, target_height=None, edges='original', overwrite=False)
    dbMedia.cropObjects(c, args)
    c.execute('SELECT x1,y1,width,height FROM objects')
    x1,y1,width,height = c.fetchone()
    self.assertEqual((x1, y1, width, height), (25, 35, 20, 40))
    c.execute('SELECT x,y FROM polygons')
    x,y = c.fetchone()
    self.assertEqual((x, y), (25, 35))



if __name__ == '__main__':
  progressbar.streams.wrap_stdout()
  logging.basicConfig (level=logging.INFO)

  unittest.main()
