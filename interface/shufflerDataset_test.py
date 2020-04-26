import os, sys, os.path as op
import random
import logging
import sqlite3
import shutil
import numpy as np
from glob import glob
import imageio
import unittest
import tempfile
from datetime import datetime, timedelta

from lib.backend.backendDb import objectField
from interface.shufflerDataset import DatasetWriter


class TestDatasetWriter(unittest.TestCase):
    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        self.writer = None

    def tearDown(self):
        shutil.rmtree(self.work_dir)

        # Close an SQLite connection, if left open.
        try:
            self.conn.close()
        except:
            pass

        # Close the DatasetWriter, if left open.
        try:
            self.writer.close()
        except:
            pass

    def test_init_needs_argument(self):
        with self.assertRaises(TypeError):
            self.writer = DatasetWriter()
        with self.assertRaises(TypeError):
            self.writer = DatasetWriter(None)

    def test_record_images(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        image_path = op.join(self.work_dir, 'images')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        self.writer = DatasetWriter(out_db_file, image_path=image_path)

        # Write one image.
        self.writer.addImage({'image': image})
        self.assertTrue(op.exists(image_path))
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        self.assertEqual(len(imagesInDir), 1)

        # Write another image.
        self.writer.addImage({'image': image})
        self.assertTrue(op.exists(image_path))
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        self.assertEqual(len(imagesInDir), 2)

    def test_record_imagevideo(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        image_path = op.join(self.work_dir, 'images.avi')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        self.writer = DatasetWriter(out_db_file,
                                    image_path=image_path,
                                    media='video')
        self.writer.addImage({'image': image})
        self.writer.addImage({'image': image})
        self.writer.close()

        self.assertTrue(op.exists(image_path))
        # FIXME: imreader.get_length() returns "inf".
        #imreader = imageio.get_reader(image_path)
        #num_frames = imreader.get_length()
        #imreader.close()
        #self.assertEqual(num_frames, 2)

    def test_record_imagefile_and_masks(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        mask_path = op.join(self.work_dir, 'masks')
        mask = np.zeros((100, 100), dtype=np.uint8)

        # Write existing images.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = op.join(self.work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)
        imageio.imwrite(op.join(image_path, 'existing2.jpg'), image)

        self.writer = DatasetWriter(out_db_file,
                                    mask_path=mask_path,
                                    rootdir=self.work_dir)

        # Write one image.
        self.writer.addImage({
            'imagefile': 'images/existing1.jpg',
            'mask': mask
        })
        self.assertTrue(op.exists(mask_path))
        masksInDir = glob(op.join(mask_path, '*.png'))
        self.assertEqual(len(masksInDir), 1)

        # Write another image.
        self.writer.addImage({
            'imagefile': 'images/existing2.jpg',
            'mask': mask
        })
        self.assertTrue(op.exists(mask_path))
        masksInDir = glob(op.join(mask_path, '*.png'))
        self.assertEqual(len(masksInDir), 2)

        # There should be no images.
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        self.assertEqual(len(imagesInDir), 2)

    def test_record_images_and_masks(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        image_path = op.join(self.work_dir, 'images')
        mask_path = op.join(self.work_dir, 'masks')
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        self.writer = DatasetWriter(out_db_file,
                                    rootdir=self.work_dir,
                                    image_path=image_path,
                                    mask_path=mask_path)

        self.writer.addImage({'image': image, 'mask': mask})
        self.writer.addImage({'image': image})
        self.assertTrue(op.exists(image_path))
        self.assertTrue(op.exists(mask_path))
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        self.assertEqual(len(imagesInDir), 2)
        masksInDir = glob(op.join(mask_path, '*.png'))
        self.assertEqual(len(masksInDir), 1)
        self.writer.close()

        self.assertTrue(op.exists(out_db_file))
        self.conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = self.conn.cursor()
        c.execute('SELECT imagefile,maskfile FROM images')
        image_entries = c.fetchall()
        self.assertEqual(len(image_entries), 2)
        self.assertEqual(image_entries[0][0], 'images/000000.jpg')
        self.assertEqual(image_entries[1][0], 'images/000001.jpg')
        self.assertEqual(image_entries[0][1], 'masks/000000.png')
        self.assertEqual(image_entries[1][1], None)

    def test_record_imagefile_maskfile_and_fields(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        timestamp = '2019-07-01 11:42:43.001000'
        timestamp_start = datetime.now()

        # Write existing images.
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image_path = op.join(self.work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)
        imageio.imwrite(op.join(image_path, 'existing2.jpg'), image)

        # Write existing masks.
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask_path = op.join(self.work_dir, 'masks')
        os.makedirs(mask_path)
        imageio.imwrite(op.join(mask_path, 'existing1.png'), mask)

        self.writer = DatasetWriter(out_db_file, rootdir=self.work_dir)
        self.writer.addImage({
            'imagefile': 'images/existing1.jpg',
            'maskfile': 'masks/existing1.png'
        })
        self.writer.addImage({
            'imagefile':
            'images/existing2.jpg',
            'timestamp':
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f'),
            'name':
            'myname',
            'score':
            0.5
        })
        self.writer.close()

        self.assertTrue(op.exists(out_db_file))
        self.conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = self.conn.cursor()
        c.execute(
            'SELECT imagefile,maskfile,width,height,timestamp,name,score FROM images'
        )
        image_entries = c.fetchall()

        # Check the first entry.
        self.assertEqual(len(image_entries), 2)
        self.assertEqual(image_entries[0][0], 'images/existing1.jpg')
        self.assertEqual(image_entries[0][1], 'masks/existing1.png')
        self.assertEqual(image_entries[0][2], 200)
        self.assertEqual(image_entries[0][3], 100)
        timestamp_written = datetime.strptime(image_entries[0][4],
                                              '%Y-%m-%d %H:%M:%S.%f')
        # The generated timestamp should be a fracetion of a second after timestamp_start.
        self.assertLess(timestamp_written - timestamp_start,
                        timedelta(seconds=10))
        self.assertEqual(image_entries[0][5], None)
        self.assertEqual(image_entries[0][6], None)

        # Check the second entry.
        self.assertEqual(image_entries[1][0], 'images/existing2.jpg')
        self.assertEqual(image_entries[1][1], None)
        self.assertEqual(image_entries[1][2], 200)
        self.assertEqual(image_entries[1][3], 100)
        self.assertEqual(image_entries[1][4], timestamp)
        self.assertEqual(image_entries[1][5], 'myname')
        self.assertEqual(image_entries[1][6], 0.5)

    def test_record_imagefile_maskfile_from_video(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        video_path = op.join(self.work_dir, 'images.avi')
        # Write a dummy video file.
        with open(video_path, 'w'):
            pass

        self.writer = DatasetWriter(out_db_file,
                                    rootdir=self.work_dir,
                                    media='video')
        self.writer.addImage({
            'imagefile': 'images.avi/000001',
            'maskfile': 'images.avi/000001',
            'width': 100,
            'height': 100
        })

        with self.assertRaises(TypeError):  # Bad "height" type.
            self.writer.addImage({
                'imagefile': 'images.avi/000001',
                'width': 100,
                'height': 100.1
            })
        with self.assertRaises(TypeError):  # Bad "width" type.
            self.writer.addImage({
                'imagefile': 'images.avi/000001',
                'width': np.ones(shape=(1, ), dtype=int),
                'height': 100.1
            })
        with self.assertRaises(KeyError):  # Need keys "height", "width".
            self.writer.addImage({'imagefile': 'images.avi/000001'})
        with self.assertRaises(ValueError):
            self.writer.addImage({'imagefile': 'nonexistant.avi/000001'})
        with self.assertRaises(ValueError):
            self.writer.addImage({
                'imagefile': 'images.avi/000001',
                'maskfile': 'nonexistant.avi/000001',
                'width': 100,
                'height': 100
            })

    def test_record_objects(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        self.writer = DatasetWriter(out_db_file)

        # Require imagefile.
        with self.assertRaises(KeyError):
            objectid = self.writer.addObject({})

        objectid1 = self.writer.addObject({
            'imagefile': 'myimagefile1',
            'x1': 10,
            'y1': 20,
            'width': 30,
            'height': 40,
            'name': 'car',
            'score': 0.5
        })
        objectid2 = self.writer.addObject({
            'imagefile': 'myimagefile2',
        })
        self.writer.close()

        self.assertTrue(op.exists(out_db_file))
        self.conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = self.conn.cursor()
        c.execute('SELECT * FROM objects')
        object_entries = c.fetchall()
        self.assertEqual(len(object_entries), 2)
        self.assertEqual(objectid1, 1)
        self.assertEqual(objectField(object_entries[0], 'objectid'), 1)
        self.assertEqual(objectField(object_entries[0], 'imagefile'),
                         'myimagefile1')
        self.assertEqual(objectField(object_entries[0], 'x1'), 10)
        self.assertEqual(objectField(object_entries[0], 'y1'), 20)
        self.assertEqual(objectField(object_entries[0], 'width'), 30)
        self.assertEqual(objectField(object_entries[0], 'height'), 40)
        self.assertEqual(objectField(object_entries[0], 'name'), 'car')
        self.assertEqual(objectField(object_entries[0], 'score'), 0.5)
        self.assertEqual(objectid2, 2)
        self.assertEqual(objectField(object_entries[1], 'objectid'), 2)
        self.assertEqual(objectField(object_entries[1], 'imagefile'),
                         'myimagefile2')

    def test_object_types(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        self.writer = DatasetWriter(out_db_file)

        with self.assertRaises(TypeError):
            self.writer.addObject({'imagefile': 'myimagefile', 'x1': 0.5})
        with self.assertRaises(TypeError):
            self.writer.addObject({
                'imagefile': 'myimagefile',
                'y1': np.zeros(shape=(1, ), dtype=int)
            })
        with self.assertRaises(TypeError):
            self.writer.addObject({
                'imagefile': 'myimagefile',
                'width': 'badtype'
            })
        with self.assertRaises(TypeError):
            self.writer.addObject({'imagefile': 'myimagefile', 'height': 0.5})
        with self.assertRaises(TypeError):
            self.writer.addObject({'imagefile': 'myimagefile', 'score': 1})

    def test_record_matches(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        self.writer = DatasetWriter(out_db_file)
        objectid1 = self.writer.addObject({'imagefile': 'myimagefile'})
        objectid2 = self.writer.addObject({'imagefile': 'myimagefile'})
        objectid3 = self.writer.addObject({'imagefile': 'myimagefile'})
        objectid4 = self.writer.addObject({'imagefile': 'myimagefile'})
        # Add a match.
        match1 = self.writer.addMatch(objectid=objectid1)
        self.writer.addMatch(objectid=objectid2, match=match1)
        self.writer.addMatch(objectid=objectid3, match=match1)
        # Add another match.
        match2 = self.writer.addMatch(objectid=objectid1)
        self.writer.addMatch(objectid=objectid4, match=match2)
        self.writer.close()

        self.assertTrue(op.exists(out_db_file))
        self.conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = self.conn.cursor()
        c.execute('SELECT objectid,match FROM matches')
        match_entries = c.fetchall()
        self.assertEqual(len(match_entries), 5)
        self.assertEqual(match_entries[0][0], objectid1)
        self.assertEqual(match_entries[1][0], objectid2)
        self.assertEqual(match_entries[2][0], objectid3)
        self.assertEqual(match_entries[3][0], objectid1)
        self.assertEqual(match_entries[4][0], objectid4)
        self.assertEqual(match_entries[0][1], match1)
        self.assertEqual(match_entries[1][1], match1)
        self.assertEqual(match_entries[2][1], match1)
        self.assertEqual(match_entries[3][1], match2)
        self.assertEqual(match_entries[4][1], match2)

    def test_context(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        # Write existing images.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = op.join(self.work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)

        with DatasetWriter(out_db_file, rootdir=self.work_dir) as self.writer:
            self.writer.addImage({'imagefile': 'images/existing1.jpg'})

        self.assertTrue(op.exists(out_db_file))
        self.conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = self.conn.cursor()
        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        self.assertEqual(len(imagefiles), 1)
        self.assertEqual(imagefiles[0][0], 'images/existing1.jpg')

    def test_failed_record_imagefile(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        self.writer = DatasetWriter(out_db_file)

        # Try to record an imagefile of an image that does not exist.
        with self.assertRaises(
                ValueError):  # Cant use FileExistsError in Python 2.
            self.writer.addImage({'imagefile': 'nonexisting.jpg'})

    def test_failed_record_maskfile(self):
        out_db_file = op.join(self.work_dir, 'out.db')
        image_path = op.join(self.work_dir, 'images')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        self.writer = DatasetWriter(out_db_file, image_path=image_path)

        # Try to record an maskfile of mask that does not exist.
        with self.assertRaises(
                ValueError):  # Cant use FileExistsError in Python 2.
            self.writer.addImage({
                'image': image,
                'maskfile': 'nonexisting.png'
            })

    def test_overwrite_db(self):
        out_db_file = op.join(self.work_dir, 'out.db')

        # Check "overwrite" for the .db file.
        self.writer = DatasetWriter(out_db_file)
        with self.assertRaises(
                ValueError):  # Cant use FileExistsError in Python 2.
            self.writer = DatasetWriter(out_db_file)
        self.writer = DatasetWriter(out_db_file, overwrite=True)

    def test_overwrite_video(self):
        out_db_file1 = op.join(self.work_dir, 'out1.db')
        out_db_file2 = op.join(self.work_dir, 'out2.db')
        out_db_file3 = op.join(self.work_dir, 'out3.db')
        out_video_path = op.join(self.work_dir, 'images.avi')

        # Check "overwrite" for the video file.
        self.writer = DatasetWriter(out_db_file1,
                                    image_path=out_video_path,
                                    media='video')
        self.writer.addImage(
            {'image': np.zeros((100, 100, 3), dtype=np.uint8)})
        self.writer.close()
        with self.assertRaises(
                ValueError):  # Cant use FileExistsError in Python 2.
            self.writer = DatasetWriter(out_db_file2,
                                        image_path=out_video_path,
                                        media='video')
        self.writer = DatasetWriter(out_db_file3,
                                    image_path=out_video_path,
                                    media='video',
                                    overwrite=True)
        self.writer.close()

    def test_overwrite_images(self):
        out_db_file1 = op.join(self.work_dir, 'out1.db')
        out_db_file2 = op.join(self.work_dir, 'out2.db')
        out_db_file3 = op.join(self.work_dir, 'out3.db')
        out_image_path = op.join(self.work_dir, 'images')

        # Check "overwrite" for the video file.
        self.writer = DatasetWriter(out_db_file1, image_path=out_image_path)
        with self.assertRaises(
                ValueError):  # Cant use FileExistsError in Python 2.
            self.writer = DatasetWriter(out_db_file2,
                                        image_path=out_image_path)
        self.writer = DatasetWriter(out_db_file3,
                                    image_path=out_image_path,
                                    overwrite=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
