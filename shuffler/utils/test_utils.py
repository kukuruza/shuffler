import os, os.path as op
import sqlite3
import shutil
import unittest
import pprint
import tempfile

from shuffler.backend import backend_db


class Test_DB(unittest.TestCase):
    ''' Implements utils to compare the result db with the expected one. '''

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
            raise TypeError('"expected" should be int, not %s' %
                            type(expected))

    def verify_that_expected_is_a_list_of_ints(self, expected):
        if not isinstance(expected, list):
            raise TypeError('"expected" should be list, not %s' %
                            type(expected))
        if len(expected) > 0 and not isinstance(expected[0], int):
            raise TypeError(
                'each element of "expected" should be int, not %s' %
                type(expected[0]))

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
            'SELECT COUNT(o.imagefile) FROM images i LEFT OUTER JOIN objects o '
            'ON i.imagefile = o.imagefile GROUP BY i.imagefile')
        actual = c.fetchall()
        expected = [(x, ) for x in expected]
        self.assertEqual(sorted(actual), sorted(expected),
                         self.summarize_db(c))

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
        expected = [(x, ) for x in expected]
        self.assertEqual(sorted(actual), sorted(expected),
                         self.summarize_db(c))

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
        expected = [(x, ) for x in expected]
        self.assertEqual(sorted(actual), sorted(expected),
                         self.summarize_db(c))

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
        expected = [(x, ) for x in expected]
        self.assertEqual(sorted(actual), sorted(expected),
                         self.summarize_db(c))


class Test_emptyDb(Test_DB):

    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        backend_db.createDb(self.conn)

    def tearDown(self):
        self.conn.close()

    def _test_table(self, cursor, table, cols_gt):

        # Test table exists.
        cursor.execute(
            'SELECT count(*) FROM sqlite_master WHERE name=? AND type="table"',
            (table, ))
        assert cursor.fetchone()[0] == 1

        # Test cols.
        cursor.execute('PRAGMA table_info(%s)' % table)
        cols_actual = [x[1] for x in cursor.fetchall()]
        self.assertEqual(set(cols_actual), set(cols_gt))

    def test_schema(self):
        cursor = self.conn.cursor()

        self._test_table(cursor, 'images', [
            'imagefile', 'maskfile', 'width', 'height', 'timestamp', 'score',
            'name'
        ])
        self._test_table(cursor, 'objects', [
            'objectid', 'imagefile', 'x1', 'y1', 'width', 'height', 'score',
            'name'
        ])
        self._test_table(cursor, 'matches', ['id', 'objectid', 'match'])
        self._test_table(cursor, 'properties',
                         ['id', 'objectid', 'key', 'value'])
        self._test_table(cursor, 'polygons',
                         ['id', 'objectid', 'x', 'y', 'name'])


class Test_carsDb(Test_DB):
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
    CARS_DB_PATH = 'testdata/cars/micro1_v5.db'
    CARS_DB_ROOTDIR = 'testdata/cars'

    def setUp(self):
        self.temp_db_path = tempfile.NamedTemporaryFile().name
        shutil.copyfile(Test_carsDb.CARS_DB_PATH, self.temp_db_path)
        self.conn = sqlite3.connect(self.temp_db_path)

    def tearDown(self):
        self.conn.close()
        if op.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
