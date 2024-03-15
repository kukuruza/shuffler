import pytest
import os, os.path as op
import sqlite3
import argparse
import tempfile
import shutil

from shuffler.backend import backend_db
from shuffler.operations import modify
from shuffler.utils import testing as testing_utils


class Test_BboxesToPolygons_SyntheticDb(testing_utils.EmptyDb):
    def test_general(self, c):
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        # Two objects in image0, no objects in image1.
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
            'VALUES ("image0",0,40,20,10,10), ("image0",1,20.5,30.5,20.5,10.5)'
        )

        args = argparse.Namespace()
        modify.bboxesToPolygons(c, args)

        self.assert_objects_count_by_imagefile(c, expected=[2, 0])
        self.assert_polygons_count_by_object(c, expected=[4, 4])

        c.execute('SELECT objectid,x,y FROM polygons ORDER BY objectid')
        actual = c.fetchall()
        expected = [(0, 40, 20), (0, 50, 20), (0, 50, 30), (0, 40, 30),
                    (1, 20.5, 30.5), (1, 41, 30.5), (1, 41, 41), (1, 20.5, 41)]
        assert set(actual) == set(expected)


class Test_PolygonsToBboxes_SyntheticDb(testing_utils.EmptyDb):
    def test_general(self, c):
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        c.execute('INSERT INTO objects(imagefile,objectid) '
                  'VALUES ("image0",0), ("image0",1), ("image0",2)')
        # A triangular polygon for object0, a rectangular one - for object1,
        # and no polygons for object2.
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES'
                  '(0,40.5,20.5), (0,50,20.5), (0,50,30), '
                  '(1,40,20), (1,50,20), (1,50,30), (1,40,30)')

        args = argparse.Namespace()
        modify.polygonsToBboxes(c, args)

        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        actual = c.fetchall()
        expected = [(0, 40.5, 20.5, 9.5, 9.5), (1, 40.0, 20.0, 10.0, 10.0),
                    (2, None, None, None, None)]
        assert set(actual) == set(expected)


class Test_Sql_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        yield c

    def test_one_command(self, c):
        args = argparse.Namespace(
            sql=['DELETE FROM images WHERE imagefile="image0"'])
        modify.sql(c, args)

        c.execute('SELECT imagefile FROM images')
        actual = c.fetchall()
        expected = [
            ("image1", ),
        ]
        assert actual == expected

    def test_multiple_commands(self, c):
        args = argparse.Namespace(sql=[
            'DELETE FROM images WHERE imagefile="image0"',
            'SELECT * FROM images'
        ])
        modify.sql(c, args)

    def test_one_command_with_multiple_statements_must_fail(self, c):
        args = argparse.Namespace(sql=[
            'DELETE FROM images WHERE imagefile="image0"; SELECT * FROM images;'
        ])
        with pytest.raises(Exception):
            modify.sql(c, args)


class Test_AddVideo_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        yield c

    def test_images_and_masks(self, c):
        args = argparse.Namespace(rootdir='testdata',
                                  image_video_path='testdata/moon/images.avi',
                                  mask_video_path='testdata/moon/masks.avi')
        modify.addVideo(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ("moon/images.avi/0", "moon/masks.avi/0", 120, 80),
            ("moon/images.avi/1", "moon/masks.avi/1", 120, 80),
            ("moon/images.avi/2", "moon/masks.avi/2", 120, 80),
        ]
        assert actual == expected

    def test_no_masks(self, c):
        args = argparse.Namespace(rootdir='testdata',
                                  image_video_path='testdata/moon/images.avi',
                                  mask_video_path=None)
        modify.addVideo(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ("moon/images.avi/0", None, 120, 80),
            ("moon/images.avi/1", None, 120, 80),
            ("moon/images.avi/2", None, 120, 80),
        ]
        assert actual == expected


class Test_AddPictures_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        yield c

    def test_images_and_masks(self, c):
        ''' Add 3 images with masks. '''
        args = argparse.Namespace(rootdir='testdata',
                                  image_pattern='testdata/moon/images/*.jpg',
                                  mask_pattern='testdata/moon/masks/*.png',
                                  width_hint=None,
                                  height_hint=None)
        modify.addPictures(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ('moon/images/000000.jpg', 'moon/masks/000000.png', 120, 80),
            ('moon/images/000001.jpg', 'moon/masks/000001.png', 120, 80),
            ('moon/images/000002.jpg', 'moon/masks/000002.png', 120, 80),
        ]
        assert actual == expected

    def test_no_masks(self, c):
        ''' Add 3 images without masks. '''
        args = argparse.Namespace(rootdir='testdata',
                                  image_pattern='testdata/moon/images/*.jpg',
                                  mask_pattern=None,
                                  width_hint=None,
                                  height_hint=None)
        modify.addPictures(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ('moon/images/000000.jpg', None, 120, 80),
            ('moon/images/000001.jpg', None, 120, 80),
            ('moon/images/000002.jpg', None, 120, 80),
        ]
        assert actual == expected

    def test_images_and_masks_with_width_hint(self, c):
        ''' Add 3 pictures with only width hint. Should still query height. '''
        args = argparse.Namespace(rootdir='testdata',
                                  image_pattern='testdata/moon/images/*.jpg',
                                  mask_pattern='testdata/moon/masks/*.png',
                                  width_hint=120,
                                  height_hint=None)
        modify.addPictures(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ('moon/images/000000.jpg', 'moon/masks/000000.png', 120, 80),
            ('moon/images/000001.jpg', 'moon/masks/000001.png', 120, 80),
            ('moon/images/000002.jpg', 'moon/masks/000002.png', 120, 80),
        ]
        assert actual == expected

    def test_images_and_masks_with_width_and_height_hint(self, c):
        ''' Add 3 pictures, with height and width hint. '''
        args = argparse.Namespace(rootdir='testdata',
                                  image_pattern='testdata/moon/images/*.jpg',
                                  mask_pattern='testdata/moon/masks/*.png',
                                  width_hint=120,
                                  height_hint=80)
        modify.addPictures(c, args)

        c.execute('SELECT imagefile,maskfile,width,height FROM images')
        actual = c.fetchall()
        expected = [
            ("image0", None, None, None),
            ("image1", None, None, None),
            ('moon/images/000000.jpg', 'moon/masks/000000.png', 120, 80),
            ('moon/images/000001.jpg', 'moon/masks/000001.png', 120, 80),
            ('moon/images/000002.jpg', 'moon/masks/000002.png', 120, 80),
        ]
        assert actual == expected


class Test_HeadImages_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        # Add 2 images, each with one object, each with one property.
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        c.execute(
            'INSERT INTO objects(objectid,imagefile) VALUES (0,"image0"), (1,"image1")'
        )
        c.execute('INSERT INTO properties(objectid) VALUES (0), (1)')
        yield c

    def test_general(self, c):
        args = argparse.Namespace(n=1)
        modify.headImages(c, args)

        c.execute('SELECT o.imagefile,o.objectid FROM images i '
                  'JOIN objects o ON i.imagefile = o.imagefile '
                  'JOIN properties p ON o.objectid = p.objectid')
        actual = c.fetchall()
        expected = [("image0", 0)]
        assert actual == expected

    def test_too_much(self, c):
        ''' When asked for more images that the db has, return all images. '''
        args = argparse.Namespace(n=5)
        modify.headImages(c, args)

        c.execute('SELECT imagefile FROM images')
        actual = c.fetchall()
        expected = [("image0", ), ("image1", )]
        assert actual == expected

    def test_invalid(self, c):
        ''' When asked for <= 0 images, raises an error. '''
        args = argparse.Namespace(n=0)
        with pytest.raises(ValueError):
            modify.headImages(c, args)

        args = argparse.Namespace(n=-5)
        with pytest.raises(ValueError):
            modify.headImages(c, args)


class Test_TailImages_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        # Add 2 images, each with one object, each with one property.
        c.execute('INSERT INTO images(imagefile) '
                  'VALUES ("image0"), ("image1")')
        c.execute('INSERT INTO objects(objectid,imagefile) '
                  'VALUES (0, "image0"), (1, "image1")')
        c.execute('INSERT INTO properties(objectid) '  # Makes line break.
                  'VALUES (0), (1)')
        yield c

    def test_general(self, c):
        args = argparse.Namespace(n=1)
        modify.tailImages(c, args)

        c.execute('SELECT o.imagefile,o.objectid FROM images i '
                  'JOIN objects o ON i.imagefile = o.imagefile '
                  'JOIN properties p ON o.objectid = p.objectid')
        actual = c.fetchall()
        expected = [("image1", 1)]
        assert actual == expected

    def test_too_much(self, c):
        ''' When asked for more images that the db has, return all images. '''
        args = argparse.Namespace(n=5)
        modify.tailImages(c, args)

        c.execute('SELECT imagefile FROM images')
        actual = c.fetchall()
        expected = [("image0", ), ("image1", )]
        assert actual == expected

    def test_invalid(self, c):
        ''' When asked for <= 0 images, raises an error. '''
        args = argparse.Namespace(n=0)
        with pytest.raises(ValueError):
            modify.tailImages(c, args)

        args = argparse.Namespace(n=-5)
        with pytest.raises(ValueError):
            modify.tailImages(c, args)


class Test_RandomNImages_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        # Add 2 images, each with one object, each with one property.
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        c.execute(
            'INSERT INTO objects(objectid,imagefile) VALUES (0,"image0"), (1,"image1")'
        )
        c.execute('INSERT INTO properties(objectid) VALUES (0), (1)')
        yield c

    def test_general(self, c):
        args = argparse.Namespace(n=1, seed=0)
        modify.randomNImages(c, args)

        c.execute('SELECT o.imagefile,o.objectid FROM images i '
                  'JOIN objects o ON i.imagefile = o.imagefile '
                  'JOIN properties p ON o.objectid = p.objectid')
        actual = c.fetchall()
        expected_option1 = [("image0", 0)]
        expected_option2 = [("image1", 1)]
        assert actual in [expected_option1, expected_option2]

    def test_too_much(self, c):
        ''' When asked for more images that the db has, return all images. '''
        args = argparse.Namespace(n=5, seed=0)
        modify.randomNImages(c, args)

        c.execute('SELECT imagefile FROM images')
        actual = c.fetchall()
        expected = [("image0", ), ("image1", )]
        assert actual == expected

    def test_invalid(self, c):
        ''' When asked for <= 0 images, raises an error. '''

        args = argparse.Namespace(n=0, seed=0)
        with pytest.raises(ValueError):
            modify.randomNImages(c, args)

        args = argparse.Namespace(n=-5)
        with pytest.raises(ValueError):
            modify.randomNImages(c, args)


class Test_ExpandObjects_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        # 2 images, one of them with 1 object.
        c.execute(
            'INSERT INTO images(imagefile) VALUES ("image0"), ("image1")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid) VALUES ("image0",0)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES'
                  '(0,40,20), (0,50,20), (0,50,30), (0,40,30)')
        yield c

    def test_polygons_only(self, c):
        args = argparse.Namespace(expand_fraction=1., target_ratio=None)
        modify.expandObjects(c, args)

        # Check that bboxes were NOT created while expanding polygons.
        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        assert c.fetchall() == [(None, None, None, None)]

        c.execute('SELECT x,y FROM polygons WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 15), (55, 15), (55, 35), (35, 35)]
        assert actual == expected

    def test_bboxes_and_polygons(self, c):
        modify.polygonsToBboxes(c, argparse.Namespace())

        args = argparse.Namespace(expand_fraction=1., target_ratio=None)
        modify.expandObjects(c, args)

        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 15, 20, 20)]
        assert actual == expected

        c.execute('SELECT x,y FROM polygons WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 15), (55, 15), (55, 35), (35, 35)]
        assert actual == expected

    def test_bboxes_only(self, c):
        modify.polygonsToBboxes(c, argparse.Namespace())
        c.execute('DELETE FROM polygons')

        args = argparse.Namespace(expand_fraction=1., target_ratio=None)
        modify.expandObjects(c, args)

        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 15, 20, 20)]
        assert actual == expected

        # Make sure polygons were not created while expanding the bbox.
        c.execute('SELECT COUNT(1) FROM polygons')
        assert c.fetchone() == (0, )

    def test_polygons_only__to_target_ratio(self, c):
        args = argparse.Namespace(expand_fraction=1., target_ratio=0.5)
        modify.expandObjects(c, args)

        # Check that bboxes were NOT created while expanding polygons.
        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        assert c.fetchall() == [(None, None, None, None)]

        c.execute('SELECT x,y FROM polygons WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 20), (55, 20), (55, 30), (35, 30)]
        assert actual == expected

    def test_bboxes_and_polygons__to_target_ratio(self, c):
        modify.polygonsToBboxes(c, argparse.Namespace())

        args = argparse.Namespace(expand_fraction=1., target_ratio=0.5)
        modify.expandObjects(c, args)

        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 20, 20, 10)]
        assert actual == expected

        c.execute('SELECT x,y FROM polygons WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 20), (55, 20), (55, 30), (35, 30)]
        assert actual == expected

    def test_bboxes_only__to_target_ratio(self, c):
        modify.polygonsToBboxes(c, argparse.Namespace())
        c.execute('DELETE FROM polygons')

        args = argparse.Namespace(expand_fraction=1., target_ratio=0.5)
        modify.expandObjects(c, args)

        c.execute('SELECT x1,y1,width,height FROM objects WHERE objectid=0')
        actual = c.fetchall()
        expected = [(35, 20, 20, 10)]
        assert actual == expected

        # Make sure polygons were not created while expanding the bbox.
        c.execute('SELECT COUNT(1) FROM polygons')
        assert c.fetchone() == (0, )


class Test_MoveMedia_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile,maskfile,width,height) VALUES '
                  '("a/b/image_c", "d/e/mask_f", 200, 100), '
                  '("a/b/image_g", NULL, 200, 100)')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("a/b/image_c", 0, 10, 20, 30, 40)')
        yield c

    def test_level1_images_and_masks(self, c):
        args = argparse.Namespace(rootdir='dummy',
                                  image_path='A/B',
                                  mask_path='D/E',
                                  level=1,
                                  verify_new_path=False,
                                  where_image='TRUE')
        modify.moveMedia(c, args)
        c.execute('SELECT imagefile,maskfile FROM images')
        entries = c.fetchall()
        assert entries[0] == ("A/B/image_c", "D/E/mask_f")
        assert entries[1] == ("A/B/image_g", None)

    def test_level1_images(self, c):
        args = argparse.Namespace(rootdir='dummy',
                                  image_path='A/B',
                                  mask_path=None,
                                  level=1,
                                  verify_new_path=False,
                                  where_image='TRUE')
        modify.moveMedia(c, args)
        c.execute('SELECT imagefile,maskfile FROM images')
        entries = c.fetchall()
        assert entries[0] == ("A/B/image_c", "d/e/mask_f")
        assert entries[1] == ("A/B/image_g", None)

    def test_level1_masks(self, c):
        args = argparse.Namespace(rootdir='dummy',
                                  image_path=None,
                                  mask_path='D/E',
                                  level=1,
                                  verify_new_path=False,
                                  where_image='TRUE')
        modify.moveMedia(c, args)
        c.execute('SELECT imagefile,maskfile FROM images')
        entries = c.fetchall()
        assert entries[0] == ("a/b/image_c", "D/E/mask_f")
        assert entries[1] == ("a/b/image_g", None)

    def test_level2_images_and_masks(self, c):
        args = argparse.Namespace(rootdir='dummy',
                                  image_path='A',
                                  mask_path='D',
                                  level=2,
                                  verify_new_path=False,
                                  where_image='TRUE')
        modify.moveMedia(c, args)
        c.execute('SELECT imagefile,maskfile FROM images')
        entries = c.fetchall()
        assert entries[0] == ("A/b/image_c", "D/e/mask_f")
        assert entries[1] == ("A/b/image_g", None)


class Test_MoveRootdir_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("a/b")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("a/b",0,10,20,30,40)')
        yield c

    def _assert_images_and_objects_consistency(self, c):
        c.execute('SELECT COUNT(1) FROM images')
        num_images = c.fetchone()[0]
        c.execute('SELECT COUNT(1) FROM images i JOIN objects o '
                  'WHERE i.imagefile = o.imagefile')
        num_same = c.fetchone()[0]
        assert num_images == num_same

    def _assert_result(self, c, rootdir, new_rootdir, expected):
        args = argparse.Namespace(rootdir=rootdir,
                                  new_rootdir=new_rootdir,
                                  verify_paths=False)
        modify.moveRootdir(c, args)
        self._assert_images_and_objects_consistency(c)
        c.execute('SELECT imagefile FROM images')
        imagefile, = c.fetchone()
        assert imagefile == expected

    def test1(self, c):
        self._assert_result(c, rootdir='.', new_rootdir='.', expected='a/b')

    def test2(self, c):
        self._assert_result(c, rootdir='.', new_rootdir='a', expected='b')

    def test3(self, c):
        self._assert_result(c, rootdir='.', new_rootdir='c', expected='../a/b')


class Test_AddDb_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def ref_conn_and_path(self):
        ref_db_path = tempfile.NamedTemporaryFile().name
        ref_conn = sqlite3.connect(ref_db_path)
        backend_db.createDb(ref_conn)
        yield ref_conn, ref_db_path
        if op.exists(ref_db_path):
            os.remove(ref_db_path)

    def fill_lowercase(self, c):
        c.execute('INSERT INTO images(imagefile) VALUES ("a"), ("b"), ("c")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,name,score) '
                  'VALUES ("a",0,10,"cat",0.1), ("b",1,20,"dog",0.2), '
                  '("c",2,30,"pig",0.3)')
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"color","gray"), (1,"breed","poodle")')
        c.execute('INSERT INTO polygons(objectid,x) VALUES (0,25), (1,35)')
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0), (1,0)')

    def fill_uppercase(self, c):
        c.execute('INSERT INTO images(imagefile) VALUES ("A"), ("B"), ("C")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,name,score) '
                  'VALUES ("A",0,10,"CAT",0.1), ("B",1,20,"DOG",0.2), '
                  '("C",2,30,"PIG",0.3)')
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"color","brown"), (1,"breed","pitbull")')
        c.execute('INSERT INTO polygons(objectid,x) VALUES (0,25), (1,35)')
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0), (1,0)')

    def test_trivial(self, conn, ref_conn_and_path):
        ''' Test on the empty database. '''
        c = conn.cursor()
        _, ref_db_path = ref_conn_and_path
        args = argparse.Namespace(rootdir='.',
                                  db_file=ref_db_path,
                                  db_rootdir=None)
        modify.addDb(c, args)

        # Verify "images" table is empty.
        c.execute('SELECT COUNT(1) FROM images')
        assert c.fetchone() == (0, )

    def test_same_images(self, conn, ref_conn_and_path):
        ''' Test on regular databases. '''
        c = conn.cursor()
        self.fill_lowercase(c)

        ref_conn, ref_db_path = ref_conn_and_path
        c_ref = ref_conn.cursor()
        self.fill_lowercase(c_ref)
        ref_conn.commit()

        args = argparse.Namespace(rootdir='.',
                                  db_file=ref_db_path,
                                  db_rootdir=None)
        modify.addDb(c, args)

        # Verify the "images" table.
        c.execute('SELECT imagefile FROM images')
        expected = [("a", ), ("b", ), ("c", )]
        assert set(c.fetchall()) == set(expected)

        # Verify the "objects" table.
        c.execute('SELECT imagefile,x1,name,score FROM objects')
        expected = [("a", 10, "cat", 0.1), ("b", 20, "dog", 0.2),
                    ("c", 30, "pig", 0.3)] * 2  # duplicate this list.
        assert c.fetchall() == expected

        # Verify the "polygons" table.
        c.execute('SELECT o.name,x FROM polygons JOIN objects o '
                  'ON polygons.objectid = o.objectid')
        expected = [("cat", 25), ("dog", 35)] * 2  # duplicate this list.
        assert c.fetchall() == expected

        # Verify the "matches" table.
        c.execute('SELECT o.name,match FROM matches JOIN objects o '
                  'ON matches.objectid = o.objectid')
        expected = [("cat", 0), ("dog", 0), ("cat", 1), ("dog", 1)]
        assert c.fetchall() == expected

        # Verify the "properties" table.
        c.execute('SELECT o.name,key,value FROM properties JOIN objects o '
                  'ON properties.objectid = o.objectid')
        expected = [("cat", "color", "gray"), ("dog", "breed", "poodle")] * 2
        assert c.fetchall() == expected

    def test_different_images(self, conn, ref_conn_and_path):
        ''' Test on regular databases. '''
        c = conn.cursor()
        self.fill_lowercase(c)

        ref_conn, ref_db_path = ref_conn_and_path
        c_ref = ref_conn.cursor()
        self.fill_uppercase(c_ref)
        ref_conn.commit()

        args = argparse.Namespace(rootdir='.',
                                  db_file=ref_db_path,
                                  db_rootdir=None)
        modify.addDb(c, args)

        # Verify the "images" table.
        c.execute('SELECT imagefile FROM images')
        expected = [("a", ), ("b", ), ("c", ), ("A", ), ("B", ), ("C", )]
        assert set(c.fetchall()) == set(expected)

        # Verify the "objects" table.
        c.execute('SELECT imagefile,x1,name,score FROM objects')
        expected = [("a", 10, "cat", 0.1), ("b", 20, "dog", 0.2),
                    ("c", 30, "pig", 0.3), ("A", 10, "CAT", 0.1),
                    ("B", 20, "DOG", 0.2), ("C", 30, "PIG", 0.3)]
        assert set(c.fetchall()) == set(expected)

    def test_db_rootdir(self, conn, ref_conn_and_path):
        c = conn.cursor()
        self.fill_lowercase(c)

        ref_conn, ref_db_path = ref_conn_and_path
        c_ref = ref_conn.cursor()
        self.fill_uppercase(c_ref)
        ref_conn.commit()

        args = argparse.Namespace(rootdir='.',
                                  db_file=ref_db_path,
                                  db_rootdir='another')
        modify.addDb(c, args)

        # Verify the "images" table.
        c.execute('SELECT imagefile FROM images')
        expected = [("a", ), ("b", ), ("c", ), ("another/A", ),
                    ("another/B", ), ("another/C", )]
        assert set(c.fetchall()) == set(expected)


class Test_SplitDb_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def fill(self, c):
        c.execute('INSERT INTO images(imagefile) VALUES ("a"), ("b")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,name,score) '
                  'VALUES ("a",0,10,"cat",0.1), ("b",1,20,"dog",0.2)')
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"color","gray"), (1,"breed","poodle")')
        c.execute('INSERT INTO polygons(objectid,x) VALUES (0,25), (1,35)')
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0),(1,0)')

    def test_empty(self, conn, work_dir):
        ''' Test on the empty database. '''
        c = conn.cursor()
        args = argparse.Namespace(rootdir='.',
                                  out_dir=work_dir,
                                  out_names=['a.db', 'b.db'],
                                  out_fractions=[0.5, 0.5],
                                  randomly=False)
        modify.splitDb(c, args)

        a_path = os.path.join(work_dir, 'a.db')
        b_path = os.path.join(work_dir, 'b.db')
        assert os.path.exists(a_path)
        assert os.path.exists(b_path)

        # Verify "images" table is empty in both splits.
        conn_a = sqlite3.connect('file:%s?mode=ro' % a_path, uri=True)
        conn_b = sqlite3.connect('file:%s?mode=ro' % b_path, uri=True)
        c_a = conn_a.cursor()
        c_b = conn_b.cursor()
        c_a.execute('SELECT COUNT(1) FROM images')
        c_b.execute('SELECT COUNT(1) FROM images')
        assert c_a.fetchone() == (0, )
        assert c_b.fetchone() == (0, )
        conn_a.close()
        conn_b.close()

    def test_regular(self, conn, work_dir):
        c = conn.cursor()
        args = argparse.Namespace(rootdir='.',
                                  out_dir=work_dir,
                                  out_names=['a.db', 'b.db'],
                                  out_fractions=[0.5, 0.5],
                                  randomly=False)
        self.fill(c)
        modify.splitDb(c, args)

        a_path = os.path.join(work_dir, 'a.db')
        b_path = os.path.join(work_dir, 'b.db')
        assert os.path.exists(a_path)
        assert os.path.exists(b_path)

        conn_a = sqlite3.connect('file:%s?mode=ro' % a_path, uri=True)
        conn_b = sqlite3.connect('file:%s?mode=ro' % b_path, uri=True)
        c_a = conn_a.cursor()
        c_b = conn_b.cursor()

        # Verify the "images" table.
        c_a.execute('SELECT imagefile FROM images')
        expected = [("a", )]
        assert set(c_a.fetchall()) == set(expected)
        c_b.execute('SELECT imagefile FROM images')
        expected = [("b", )]
        assert set(c_b.fetchall()) == set(expected)

        # Verify the "objects" table.
        c_a.execute('SELECT imagefile,x1,name,score FROM objects')
        expected = [("a", 10, "cat", 0.1)]
        assert c_a.fetchall() == expected
        c_b.execute('SELECT imagefile,x1,name,score FROM objects')
        expected = [("b", 20, "dog", 0.2)]
        assert c_b.fetchall() == expected

        # Verify the "polygons" table.
        c_a.execute('SELECT o.name,x FROM polygons JOIN objects o '
                    'ON polygons.objectid = o.objectid')
        expected = [("cat", 25)]
        assert c_a.fetchall() == expected
        c_b.execute('SELECT o.name,x FROM polygons JOIN objects o '
                    'ON polygons.objectid = o.objectid')
        expected = [("dog", 35)]
        assert c_b.fetchall() == expected

        # Verify the "matches" table.
        c_a.execute('SELECT o.name FROM matches JOIN objects o '
                    'ON matches.objectid = o.objectid')
        expected = [("cat", )]
        assert c_a.fetchall() == expected
        c_b.execute('SELECT o.name FROM matches JOIN objects o '
                    'ON matches.objectid = o.objectid')
        expected = [("dog", )]
        assert c_b.fetchall() == expected

        # Verify the "properties" table.
        c_a.execute('SELECT o.name,key,value FROM properties JOIN objects o '
                    'ON properties.objectid = o.objectid')
        expected = [("cat", "color", "gray")]
        assert c_a.fetchall() == expected
        c_b.execute('SELECT o.name,key,value FROM properties JOIN objects o '
                    'ON properties.objectid = o.objectid')
        expected = [("dog", "breed", "poodle")]
        assert c_b.fetchall() == expected

        conn_a.close()
        conn_b.close()


class Test_MergeIntersectingObjects_SyntheticDb(testing_utils.EmptyDb):
    def _fill(self, c):
        # "cat", "dog", and "pika" intersect.
        # "pig" and "goat" would too but "pig" is in another image and "goat" has no imagefile.
        # "sheep" does not intersect with them.
        c.execute('INSERT INTO images(imagefile) VALUES ("a"), ("b")')
        c.execute(
            'INSERT INTO objects(imagefile,objectid,x1,y1,width,height,name,score) VALUES '
            '("a",  0, 10,10,10,10, "cat",   0.1), '
            '("a",  1, 10,15,10,10, "dog",   null), '
            #            '("a",  2,15,10,10,10, "pika",  0.2), '
            '("a",  3, 30,10,10,10, "sheep", 1.0), '
            '("b",  4, 10,10,10,10, "pig",   1.0), '
            '(null, 5, 10,10,10,10, "goat",  1.0) ')
        c.execute(
            'INSERT INTO properties(objectid,key,value) '
            'VALUES (0,"says","miaw"), (1,"breed","poodle"), (3,"says","me")')
        # "cat" and "dog" match (this match should dissapear).
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0), (1,0)')
        # "cat", "dog" and "sheep" match (this match should stay,
        # except "cat" and "dog" objects will be merged).
        c.execute(
            'INSERT INTO matches(objectid,match) VALUES (1,1), (3,1), (0,1)')
        # "pig" is matched only with itself.
        c.execute('INSERT INTO matches(objectid,match) VALUES (4,2)')

    def _replace_dog_bbox_with_polygon(self, c):
        ''' Removes the bounding box of the "cat", and inserts the polygon. '''

        dog_id = 1  # objectid of "dog".
        c.execute('UPDATE objects SET x1=NULL WHERE objectid = ?', (dog_id, ))
        c.execute(
            'INSERT INTO polygons(objectid,x,y) '
            'VALUES (?, 10,15), (?, 10,25), (?, 20,25), (?, 20,15)',
            (dog_id, dog_id, dog_id, dog_id))

    def _cast_float_xy_polygons(self, polygons):
        return [(objectid, float(x), float(y)) for objectid, x, y in polygons]

    def test_bboxes_only(self, c):
        self._fill(c)

        args = argparse.Namespace(IoU_threshold=0.1,
                                  where_image='TRUE',
                                  where_object='TRUE')
        modify.mergeIntersectingObjects(c, args)

        # Verify the "objects" table.
        c.execute(
            'SELECT imagefile,objectid,x1,y1,width,height,name,score FROM objects'
        )
        expected = [("a", 0, 10.0, 10.0, 10.0, 15.0, "cat", 0.1),
                    ("a", 3, 30.0, 10.0, 10.0, 10.0, "sheep", 1.0),
                    ("b", 4, 10.0, 10.0, 10.0, 10.0, "pig", 1.0),
                    (None, 5, 10.0, 10.0, 10.0, 10.0, "goat", 1.0)]
        assert set(c.fetchall()) == set(expected)

        # Verify the "polygons" table.
        c.execute('SELECT objectid,x,y FROM polygons')
        expected = [(0, 10, 10), (0, 10, 20), (0, 20, 20), (0, 20, 10),
                    (0, 10, 15), (0, 10, 25), (0, 20, 25), (0, 20, 15),
                    (3, 30, 10), (3, 40, 10), (3, 40, 20), (3, 30, 20),
                    (4, 10, 10), (4, 10, 20), (4, 20, 20), (4, 20, 10),
                    (5, 10, 10), (5, 10, 20), (5, 20, 20), (5, 20, 10)]
        assert set(c.fetchall()) == set(self._cast_float_xy_polygons(expected))

        # Verify the "properties" table.
        c.execute('SELECT objectid,key,value FROM properties')
        expected = [(0, "says", "miaw"), (0, "breed", "poodle"),
                    (0, "is_merged", "1"), (0, "merged_name", "cat"),
                    (0, "merged_name", "dog"), (3, "says", "me")]
        assert set(c.fetchall()) == set(expected)

        # Verify the "matches" table.
        c.execute('SELECT objectid,match FROM matches')
        expected = [(0, 1), (3, 1), (4, 2)]
        assert set(c.fetchall()) == set(expected)

    def test_bboxes_and_polygons(self, c):
        self._fill(c)
        self._replace_dog_bbox_with_polygon(c)  # <- differs from bboxes only.

        args = argparse.Namespace(IoU_threshold=0.1,
                                  where_image='TRUE',
                                  where_object='TRUE')
        modify.mergeIntersectingObjects(c, args)

        # Verify the "objects" table.
        c.execute(
            'SELECT imagefile,objectid,x1,y1,width,height,name,score FROM objects'
        )
        expected = [("a", 0, 10.0, 10.0, 10.0, 15.0, "cat", 0.1),
                    ("a", 3, 30.0, 10.0, 10.0, 10.0, "sheep", 1.0),
                    ("b", 4, 10.0, 10.0, 10.0, 10.0, "pig", 1.0),
                    (None, 5, 10.0, 10.0, 10.0, 10.0, "goat", 1.0)]
        assert set(c.fetchall()) == set(expected)

        # Verify the "polygons" table.
        c.execute('SELECT objectid,x,y FROM polygons')
        expected = [(0, 10, 10), (0, 10, 20), (0, 20, 20), (0, 20, 10),
                    (0, 10, 15), (0, 10, 25), (0, 20, 25), (0, 20, 15),
                    (3, 30, 10), (3, 40, 10), (3, 40, 20), (3, 30, 20),
                    (4, 10, 10), (4, 10, 20), (4, 20, 20), (4, 20, 10),
                    (5, 10, 10), (5, 10, 20), (5, 20, 20), (5, 20, 10)]
        assert set(c.fetchall()) == set(self._cast_float_xy_polygons(expected))


class Test_PropertyToObjectsField_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("a"), ("b"), ("c")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,name,score) '
                  'VALUES ("a",0,10,"cat",0.1), ("b",1,20,"dog",0.2), '
                  '("c",2,30,"pig",0.3)')
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"color","gray"), (1,"breed","poodle")')
        c.execute('INSERT INTO polygons(objectid,x) VALUES (0,25), (1,35)')
        c.execute('INSERT INTO matches(objectid,match) VALUES (0,0), (1,0)')
        yield c

    def test_trivial(self, conn):
        ''' Test on the empty database. '''
        c = conn.cursor()
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","dummy")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='name',
                                  properties_key='newval')
        modify.propertyToObjectsField(c, args)

    def test_bad_field(self, c):
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='bad_field',
                                  properties_key='newval')
        with pytest.raises(ValueError):
            modify.propertyToObjectsField(c, args)

    def test_absent_key(self, c):
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='absent_key')
        with pytest.raises(ValueError):
            modify.propertyToObjectsField(c, args)

    def test_objectid_field(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","2"), (2,"newval","6")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        modify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT imagefile,objectid FROM objects')
        expected = [("a", 2), ("b", 1), ("c", 6)]
        assert set(c.fetchall()) == set(expected)

        # Verify the "polygons" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,x FROM polygons')
        expected = [(2, 25), (1, 35)]
        assert set(c.fetchall()) == set(expected)

        # Verify the "matches" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,match FROM matches')
        expected = [(2, 0), (1, 0)]
        assert set(c.fetchall()) == set(expected)

        # Verify the "properties" table (the newval 0 is replaced with 2).
        c.execute('SELECT objectid,key,value FROM properties')
        expected = [(2, "color", "gray"), (1, "breed", "poodle"),
                    (2, "newval", "2"), (6, "newval", "6")]
        assert set(c.fetchall()) == set(expected)

    def test_objectid_field__non_unique_values(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","2"), (1,"newval","2")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        with pytest.raises(Exception):
            modify.propertyToObjectsField(c, args)

    def test_objectid_field__value_matches_not_updated_entry(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","1")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='objectid',
                                  properties_key='newval')
        with pytest.raises(Exception):
            modify.propertyToObjectsField(c, args)

    def test_name_field(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","sheep")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='name',
                                  properties_key='newval')
        modify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,name FROM objects')
        expected = [(0, "sheep"), (1, "dog"), (2, "pig")]
        assert set(c.fetchall()) == set(expected)

    def test_x1_field(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","50")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='x1',
                                  properties_key='newval')
        modify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,x1 FROM objects')
        expected = [(0, 50), (1, 20), (2, 30)]
        assert set(c.fetchall()) == set(expected)

    def test_score_field(self, c):
        c.execute('INSERT INTO properties(objectid,key,value) '
                  'VALUES (0,"newval","0.5")')
        args = argparse.Namespace(rootdir='.',
                                  target_objects_field='score',
                                  properties_key='newval')
        modify.propertyToObjectsField(c, args)

        # Verify the "objects" table.
        c.execute('SELECT objectid,score FROM objects')
        expected = [(0, 0.5), (1, 0.2), (2, 0.3)]
        assert set(c.fetchall()) == set(expected)


class Test_SyncPolygonIdsWithDb_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def ref_conn_and_path(self):
        ref_db_path = tempfile.NamedTemporaryFile().name
        ref_conn = sqlite3.connect(ref_db_path)
        backend_db.createDb(ref_conn)
        yield ref_conn, ref_db_path
        if op.exists(ref_db_path):
            os.remove(ref_db_path)

    def _insert_polygons_value(self, c, vals):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals:  a tuple with 5 numbers (id,objectid,x,y,name).
        '''
        s = 'polygons(id,objectid,x,y,name)'
        if len(vals):
            c.execute('INSERT INTO %s VALUES %s' %
                      (s, testing_utils.insertValuesToStr(vals)))

    def _insert_polygons_value_to_ref(self, ref_conn, vals_ref):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals_ref:  a tuple with 5 numbers (id,objectid,x,y,name).
        '''
        c_ref = ref_conn.cursor()
        s = 'polygons(id,objectid,x,y,name)'
        if len(vals_ref):
            c_ref.execute('INSERT INTO %s VALUES %s' %
                          (s, testing_utils.insertValuesToStr(vals_ref)))
        ref_conn.commit()
        ref_conn.close()

    def test_empty(self, c, ref_conn_and_path):
        ref_conn, ref_db_path = ref_conn_and_path
        vals_ref = [(1, 1, 10, 20, 'name')]
        self._insert_polygons_value(c, [])
        self._insert_polygons_value_to_ref(ref_conn, vals_ref)
        args = argparse.Namespace(ref_db_file=ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        modify.syncPolygonIdsWithDb(c, args)
        c.execute('SELECT id,objectid,x,y,name FROM polygons')
        assert c.fetchall() == []

    def test_no_update_because_of_different_object(self, c, ref_conn_and_path):
        ''' No update is expected because the objects mismatch. '''
        ref_conn, ref_db_path = ref_conn_and_path
        vals = [(1, 1, 10, 20, 'name1'), (2, 1, 10, 20, 'name2')]
        vals_ref = [(1, 2, 10, 20, 'name1'), (2, 2, 10, 20, 'name2')]
        self._insert_polygons_value(c, vals)
        self._insert_polygons_value_to_ref(ref_conn, vals_ref)
        args = argparse.Namespace(ref_db_file=ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        modify.syncPolygonIdsWithDb(c, args)

        # Not checking ids here, because unmatched polygons points get new ids.
        vals_expected = [(1, 10, 20, 'name1'), (1, 10, 20, 'name2')]
        c.execute('SELECT objectid,x,y,name FROM polygons')
        assert c.fetchall() == vals_expected

    def test_all_match__ignore_name(self, c, ref_conn_and_path):
        '''
        All points match. Objectids is different. Ignore name.
        '''
        # Will need to reverse ids.
        ref_conn, ref_db_path = ref_conn_and_path
        vals = [(4, 2, 10.5, 0.5, None), (3, 1, 20.5, 0.5, None),
                (2, 1, 10.5, 0.5, None), (1, 2, 20.5, 0.5, None)]
        vals_ref = [(1, 2, 10, 0, 'name1'), (2, 1, 20, 0, 'name2'),
                    (3, 1, 10, 0, 'name3'), (4, 2, 20, 0, 'name4')]
        self._insert_polygons_value(c, vals)
        self._insert_polygons_value_to_ref(ref_conn, vals_ref)
        args = argparse.Namespace(ref_db_file=ref_db_path,
                                  epsilon=1.,
                                  ignore_name=True)
        modify.syncPolygonIdsWithDb(c, args)

        vals_expected = [(1, 2, 10.5, 0.5, None), (2, 1, 20.5, 0.5, None),
                         (3, 1, 10.5, 0.5, None), (4, 2, 20.5, 0.5, None)]
        c.execute('SELECT id,objectid,x,y,name FROM polygons ORDER BY id ASC')
        assert c.fetchall() == vals_expected

    def test_all_match__match_name(self, c, ref_conn_and_path):
        '''
        All points match. Objectids and coordinates is the same. Use name.
        '''
        # Will need to reverse ids.
        ref_conn, ref_db_path = ref_conn_and_path
        vals = [(4, 1, 10.5, 0.5, 'name4'), (3, 1, 10.5, 0.5, 'name3'),
                (2, 1, 10.5, 0.5, None), (1, 1, 10.5, 0.5, 'name1')]
        vals_ref = [(1, 1, 10, 0, 'name1'), (2, 1, 10, 0, None),
                    (3, 1, 10, 0, 'name3'), (4, 1, 10, 0, 'name4')]
        self._insert_polygons_value(c, vals)
        self._insert_polygons_value_to_ref(ref_conn, vals_ref)
        args = argparse.Namespace(ref_db_file=ref_db_path,
                                  epsilon=1.,
                                  ignore_name=False)
        modify.syncPolygonIdsWithDb(c, args)

        vals_expected = [(1, 1, 10.5, 0.5, 'name1'), (2, 1, 10.5, 0.5, None),
                         (3, 1, 10.5, 0.5, 'name3'),
                         (4, 1, 10.5, 0.5, 'name4')]
        c.execute('SELECT id,objectid,x,y,name FROM polygons ORDER BY id ASC')
        assert c.fetchall() == vals_expected


class Test_SyncRoundedCoordinatesWithDb_SyntheticDb(testing_utils.EmptyDb):
    @pytest.fixture()
    def ref_conn_and_path(self):
        ref_db_path = tempfile.NamedTemporaryFile().name
        ref_conn = sqlite3.connect(ref_db_path)
        backend_db.createDb(ref_conn)
        yield ref_conn, ref_db_path
        if op.exists(ref_db_path):
            os.remove(ref_db_path)

    def _insert_objects_value(self, c, vals):
        '''
        Insert values into 'objects' table of the active and ref dbs.
        Args:
          vals:  a tuple with 5 numbes (objectid,x1,y1,width,height).
        '''
        s = 'objects(objectid,x1,y1,width,height)'
        if len(vals):
            c.execute('INSERT INTO %s VALUES %s' %
                      (s, testing_utils.insertValuesToStr(vals)))

    def _insert_objects_value_to_ref(self, ref_conn, vals_ref):
        '''
        Insert values into 'objects' table of the active and ref dbs.
        Args:
          vals_ref:  a tuple with 5 numbes (objectid,x1,y1,width,height).
        '''
        c_ref = ref_conn.cursor()
        s = 'objects(objectid,x1,y1,width,height)'
        if len(vals_ref):
            c_ref.execute('INSERT INTO %s VALUES %s' %
                          (s, testing_utils.insertValuesToStr(vals_ref)))
        ref_conn.commit()
        ref_conn.close()

    def _insert_polygons_value(self, c, vals):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals:  a tuple with 3 numbers (id,x,y).
        '''
        if len(vals):
            c.execute('INSERT INTO polygons(id,x,y) VALUES %s' %
                      testing_utils.insertValuesToStr(vals))

    def _insert_polygons_value_to_ref(self, ref_conn, vals_ref):
        ''' Insert values into 'polygons' table of the active and ref dbs.
        Args:
          vals_ref:  a tuple with 3 numbers (id,x,y).
        '''
        c_ref = ref_conn.cursor()
        if len(vals_ref):
            c_ref.execute('INSERT INTO polygons(id,x,y) VALUES %s' %
                          testing_utils.insertValuesToStr(vals_ref))
        ref_conn.commit()
        ref_conn.close()

    def test_objects(self, c, ref_conn_and_path):
        '''
        Sync x1, x2, y1, or y2  in 'objects', whichever has changed.
        E.g.:
          - if x2 = x1 + width has changed, update width.
          - if x1 has changed, update both x1 and width.
        '''
        ref_conn, ref_db_path = ref_conn_and_path
        #  Objects with identical coordinates.
        vals = [(0, 10, 20, 30, 40), (1, 10, 20, 30, 40), (2, 10, 20, 30, 40),
                (3, 10, 20, 30, 40), (4, 10, 20, 30, 40), (5, 10, 20, 30, 40),
                (6, 10, 20, 30, 40), (7, 10, 20, 30, 40), (8, 10, 20, 30, 40),
                (9, 10, 20, 30, 40)]
        vals_ref = [
            (0, 10, 20, 30, 40),  # No change.
            (1, 10.1, 20, 29.9, 40),  # Changed a bit only x1 (x2 is fixed).
            (2, 10, 20.2, 30, 39.8),  # Changed a bit only y1 (y2 is fixed).
            (3, 10, 20, 30.3, 40),  # Changed a bit only x2 (via width).
            (4, 10, 20, 30, 40.4),  # Changed a bit only y2 (via height).
            (5, 10.1, 20, 35, 40),  # x1 changed a bit, but x2 changed a lot.
            (6, 10, 20.1, 30, 45),  # y1 changed a bit, but y2 changed a lot.
            (7, 10.1, 20.2, 30, 40),  # x1,y1,x2,y2 changed a bit.
            (8, 15, 25, 35, 45),  # x1,y1,x2,y2 changed a lot.
            # objectid=9 is new in active db (missing from ref_db).
            (10, 10, 20, 30, 40),  # missing from active db.
        ]
        self._insert_objects_value(c, vals)
        self._insert_objects_value_to_ref(ref_conn, vals_ref)
        modify.syncRoundedCoordinatesWithDb(
            c, argparse.Namespace(ref_db_file=ref_db_path, epsilon=1.))

        vals_expected = [
            (0, 10, 20, 30, 40),  # No change.
            (1, 10.1, 20, 29.9, 40
             ),  # x1 is taken from ref_db, width is computed to keep x2 fixed.
            (2, 10, 20.2, 30, 39.8
             ),  # y1 is taken from ref_db, height is computed to keep y2 fixed
            (3, 10, 20, 30.3, 40),  # only width is taken from ref_db.
            (4, 10, 20, 30, 40.4),  # only height is taken from ref_db.
            (5, 10.1, 20, 35, 40
             ),  # x1 is taken from ref_db, width is computed to keep x2 fixed.
            (6, 10, 20.1, 30, 45
             ),  # y1 is taken from ref_db, height is computed to keep y2 fixed
            (7, 10.1, 20.2, 30, 40),  # all are taken from ref_db.
            (8, 10, 20, 30, 40),  # all are fixed.
            (9, 10, 20, 30, 40),  # all are taken from active db since it's new
        ]
        c.execute('SELECT objectid,x1,y1,width,height FROM objects')
        entries = c.fetchall()
        assert entries == vals_expected

    def test_polygons(self, c, ref_conn_and_path):
        ''' Sync x or y in 'polygons', whichever has changed. '''
        ref_conn, ref_db_path = ref_conn_and_path
        vals = [(0, 10, 20), (1, 10, 20), (2, 10, 20), (3, 10, 20),
                (4, 10, 20), (5, 10, 20)]
        vals_ref = [
            (0, 10, 20),  # No change.
            (1, 10.1, 20),  # Changed a bit only x.
            (2, 10, 20.2),  # Changed a bit only y.
            (3, 10.1, 20.2),  # x,y changed a bit.
            (4, 15, 25),  # x,y changed a lot.
            # objectid=5 is new in active db (missing from ref_db).
            (6, 10, 20),  # missing from active db.
        ]
        self._insert_polygons_value(c, vals)
        self._insert_polygons_value_to_ref(ref_conn, vals_ref)
        modify.syncRoundedCoordinatesWithDb(
            c, argparse.Namespace(ref_db_file=ref_db_path, epsilon=1.))

        vals_expected = [
            (0, 10, 20),  # No change.
            (1, 10.1, 20),  # x is taken from ref_db.
            (2, 10, 20.2),  # y is taken from ref_db.
            (3, 10.1, 20.2),  # all are taken from ref_db.
            (4, 10, 20),  # all are fixed.
            (5, 10, 20),  # all are taken from active db since it's new
        ]
        c.execute('SELECT id,x,y FROM polygons')
        entries = c.fetchall()
        assert entries == vals_expected


class Test_RevertObjectTransforms_SyntheticDb(testing_utils.EmptyDb):
    # The original bbox (x1, y1, width, height).
    original_bbox_gt = (20, 60, 5, 20)
    # The original polygon [(x, y)] * N.
    original_polygon_gt = [(20, 60), (25, 60), (25, 80), (20, 80)]

    @pytest.fixture()
    def c(self, conn):
        c = conn.cursor()
        c.execute('INSERT INTO images(imagefile) VALUES ("image0")')
        c.execute('INSERT INTO objects(imagefile,objectid,x1,y1,width,height) '
                  'VALUES ("image0",0,45,25,10,10)')
        c.execute('INSERT INTO polygons(objectid,x,y) VALUES '
                  '(0,45,25), (0,55,25), (0,55,35), (0,45,35)')
        # transform = [[2., 0.,   5.]
        #              [0., 0.5, -5.]]
        c.execute('INSERT INTO properties(objectid,key,value) VALUES '
                  '(0,"kx","2"), (0,"ky","0.5"), (0,"bx","5"), (0,"by","-5.")')
        yield c

    def test_general(self, c):
        args = argparse.Namespace(rootdir='.')
        modify.revertObjectTransforms(c, args)
        # Check bbox.
        c.execute('SELECT x1,y1,width,height FROM objects')
        original_bboxes = c.fetchall()
        assert len(original_bboxes) == 1
        original_bbox = original_bboxes[0]
        assert self.original_bbox_gt == original_bbox
        # Check polygons.
        c.execute('SELECT x,y FROM polygons')
        original_polygon = c.fetchall()
        assert len(original_polygon) == 4
        assert self.original_polygon_gt == original_polygon
