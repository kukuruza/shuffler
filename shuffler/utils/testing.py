import pytest
import os, os.path as op
import sqlite3
import shutil
import pprint
import tempfile

from shuffler.backend import backend_db


def insertValuesToStr(vals):
    '''
    Makes a string from INSERT values.
    Args:
        vals:  a list of tuples, e.g. [(0, 1), (1, 2)].
    Returns:
        a string, e.g. '(0, 1), (1, 2)'.
    '''
    vals_str = []
    for val in vals:
        val_str = ','.join(
            ['"%s"' % x if x is not None else 'NULL' for x in val])
        vals_str.append(val_str)
    s = ', '.join(['(%s)' % x for x in vals_str])
    return s


class _BaseDB:
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
        assert c.fetchone()[0] == expected, self.summarize_db(c)

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
        assert sorted(actual) == sorted(expected), self.summarize_db(c)

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
        assert sorted(actual) == sorted(expected), self.summarize_db(c)

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
        assert sorted(actual) == sorted(expected), self.summarize_db(c)

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
        assert sorted(actual) == sorted(expected), self.summarize_db(c)


class EmptyDb(_BaseDB):
    @pytest.fixture()
    def conn(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn
        conn.close()

    @pytest.fixture()
    def c(self):
        conn = sqlite3.connect(':memory:')
        backend_db.createDb(conn)
        yield conn.cursor()
        conn.close()


class CarsDb(_BaseDB):
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

    @pytest.fixture()
    def c(self):
        temp_db_path = tempfile.NamedTemporaryFile().name
        shutil.copyfile(self.CARS_DB_PATH, temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        yield conn.cursor()
        conn.close()
        if op.exists(temp_db_path):
            os.remove(temp_db_path)
