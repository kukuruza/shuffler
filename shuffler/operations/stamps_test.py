import pytest
import os, os.path as op
import sqlite3
import argparse
import tempfile
import shutil
import simplejson as json

from shuffler.backend import backend_db
from shuffler.operations import stamps
from shuffler.utils import testing as testing_utils


class Test_extractNumberIntoProperty(testing_utils.EmptyDb):
    def test_one_number_middle(self, c):
        # Repeated twice.
        c.execute("INSERT INTO objects(name) VALUES ('my123number')")
        c.execute("INSERT INTO objects(name) VALUES ('my123number')")
        args = argparse.Namespace(property='number')
        stamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        assert len(names) == 2  # repeated twice
        for name, in names:
            assert name == "mynumber"
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        assert len(entries) == 2  # repeated twice
        for key, value in entries:
            assert key == "number"
            assert value == "123"

    def test_one_on_end(self, c):
        c.execute("INSERT INTO objects(name) VALUES ('mynumber123')")
        args = argparse.Namespace(property='number')
        stamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        assert len(names) == 1
        assert names[0][0] == "mynumber"
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        assert len(entries) == 1
        for key, value in entries:
            assert key == "number"
            assert value == "123"

    def test_one_on_beginning(self, c):
        c.execute("INSERT INTO objects(name) VALUES ('123mynumber')")
        args = argparse.Namespace(property='number')
        stamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        assert len(names) == 1
        assert names[0][0] == "mynumber"
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        assert len(entries) == 1
        for key, value in entries:
            assert key == "number"
            assert value == "123"

    def test_two_numbers(self, c):
        c.execute("INSERT INTO objects(name) VALUES ('my123num456ber')")
        args = argparse.Namespace(property='number')
        stamps.extractNumberIntoProperty(c, args)
        # Check names.
        c.execute("SELECT name FROM objects")
        names = c.fetchall()
        assert len(names) == 1
        assert names[0][0] == "mynumber"
        # Check properties.
        c.execute("SELECT key,value FROM properties")
        entries = c.fetchall()
        keys = [e[0] for e in entries]
        values = [e[1] for e in entries]
        assert len(entries) == 2
        assert keys[0] == "number"
        assert values[0] in ["123", "456"]
        assert keys[1] == "number"
        assert values[1] in ["123", "456"]
        assert values[0] != values[1]


class Test_syncImagesWithDb_synthetic(testing_utils.EmptyDb):
    @pytest.fixture()
    def ref_db_path_and_imagefiles(self):
        ref_db_path = tempfile.NamedTemporaryFile().name
        ref_conn = sqlite3.connect(ref_db_path)
        backend_db.createDb(ref_conn)

        ref_imagefiles = ['ref_image0', 'ref_image1']
        c_ref = ref_conn.cursor()
        c_ref.execute("INSERT INTO images(imagefile) VALUES ('ref_image0')")
        c_ref.execute("INSERT INTO images(imagefile) VALUES ('ref_image1')")
        c_ref.execute("INSERT INTO objects(objectid,imagefile,name) "
                      "VALUES (0,'ref_image0','new')")
        c_ref.execute("INSERT INTO objects(objectid,imagefile,name) "
                      "VALUES (1,'ref_image1','new')")
        ref_conn.commit()

        yield ref_db_path, ref_imagefiles
        if op.exists(ref_db_path):
            os.remove(ref_db_path)

    def test_general(self, c, ref_db_path_and_imagefiles):
        # Insert an object and an image that is there in the ref_db_file.
        c.execute("INSERT INTO images(imagefile) VALUES ('image0')")
        c.execute("INSERT INTO objects(objectid,imagefile,name) "
                  "VALUES (0,'image0','new')")
        # Run the function.
        ref_db_path, ref_imagefiles = ref_db_path_and_imagefiles
        args = argparse.Namespace(ref_db_file=ref_db_path)
        stamps.syncImagesWithDb(c, args)
        # Test images.
        c.execute("SELECT imagefile FROM images")
        imagefiles = [entry[0] for entry in c.fetchall()]
        assert imagefiles == ref_imagefiles
        # Test objects.
        c.execute("SELECT objectid,name FROM objects")
        object_entries = c.fetchall()
        assert object_entries == [(0, 'new')]

    def test_objectNotFound(self, c, ref_db_path_and_imagefiles):
        c.execute("INSERT INTO images(imagefile) VALUES ('image0')")
        c.execute("INSERT INTO objects(objectid,imagefile,name) "
                  "VALUES (2,'image0','new')")
        # objectid=2 is not in ref_db_file.
        ref_db_path, _ = ref_db_path_and_imagefiles
        args = argparse.Namespace(ref_db_file=ref_db_path)
        with pytest.raises(ValueError):
            stamps.syncImagesWithDb(c, args)


class Test_getTop1Name(testing_utils.EmptyDb):
    def test_general(self, c):
        c.execute("INSERT INTO objects(objectid,imagefile,name) VALUES "
                  "(0, 'image0', 'okay'), "
                  "(1, 'image0', 'wood / stone'), "
                  "(2, 'image0', 'cat / dog / sheep')")
        # Run the function.
        stamps.getTop1Name(c, argparse.Namespace())
        c.execute("SELECT name FROM objects")
        names = [name[0] for name in c.fetchall()]
        assert names == ['okay', 'wood', 'cat']


class Test_setNumStampOccurancies(testing_utils.EmptyDb):
    def test_insert_and_update(self, c):
        # The 1st time (check insert).
        c.execute("INSERT INTO objects(name) VALUES ('cat'), ('dog'), ('cat')")
        stamps.setNumStampOccurancies(c, argparse.Namespace())
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="num_instances"')
        values = set(c.fetchall())
        # 2 for cats and 1 for dog.
        assert values == set([('cat', '2')] * 2 + [('dog', '1')] * 1)

        # The 2nd time (check update).
        c.execute('INSERT INTO objects(name) VALUES ("cat"), ("dog"), ("cat")')
        stamps.setNumStampOccurancies(c, argparse.Namespace())
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="num_instances"')
        values = set(c.fetchall())
        # 4 for cats and 2 for dog.
        assert values == set([('cat', '4')] * 4 + [('dog', '2')] * 2)


class Test_encodeNames(testing_utils.CarsDb):
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def test_WriteAndReadEncodingFile(self, c, work_dir):
        # Add the special case.
        c.execute("INSERT INTO objects(name) VALUES ('??+'), ('??+'), ('??+')")

        encoding_json_file = op.join(work_dir, 'encoding.json')

        # Encode with writing the encoding file.
        stamps.encodeNames(
            c,
            argparse.Namespace(out_encoding_json_file=encoding_json_file,
                               in_encoding_json_file=None))

        # Test the database.
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="name_id"')
        entries = c.fetchall()
        assert set(entries) == set([('bus', '1'), ('car', '0'), ('??+', '-1')])

        # Test the encoding file.
        assert op.exists(encoding_json_file)
        encoding = json.load(open(encoding_json_file))
        assert 'bus' in encoding
        assert len(encoding) == 3, encoding
        assert encoding['car'] == 0, encoding
        assert encoding['bus'] == 1, encoding
        assert encoding['??+'] == -1, encoding

        # Add the class which was not there during writing the encoding file.
        c.execute("INSERT INTO objects(name) VALUES ('dog'), ('dog')")

        # Encode with reading the encoding file.
        stamps.encodeNames(
            c,
            argparse.Namespace(in_encoding_json_file=encoding_json_file,
                               out_encoding_json_file=None))

        # Test the database.
        c.execute('SELECT o.name,value FROM objects o INNER JOIN properties p '
                  ' ON o.objectid == p.objectid WHERE key="name_id"')
        entries = c.fetchall()
        assert set(entries) == set([('bus', '1'), ('car', '0'), ('??+', '-1'),
                                    ('dog', '-1')])
