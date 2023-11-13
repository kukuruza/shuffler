import pytest
import os, os.path as op
import sqlite3
import shutil
import numpy as np
from glob import glob
import imageio
import tempfile
from datetime import datetime, timedelta

from shuffler.backend.backend_db import objectField
from shuffler.interface.shuffler_dataset import DatasetWriter


class TestDatasetWriter:
    @pytest.fixture()
    def work_dir(self):
        work_dir = tempfile.mkdtemp()
        yield work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    def test_init_needs_argument(self):
        with pytest.raises(TypeError):
            DatasetWriter()
        with pytest.raises(TypeError):
            DatasetWriter(None)

    def test_record_images(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        image_path = op.join(work_dir, 'images')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        writer = DatasetWriter(out_db_file, image_path=image_path)

        # Write one image.
        writer.addImage({'image': image})
        assert op.exists(image_path)
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        assert len(imagesInDir) == 1, len(imagesInDir)

        # Write another image.
        writer.addImage({'image': image})
        assert op.exists(image_path)
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        assert len(imagesInDir) == 2, len(imagesInDir)

    def test_record_imagevideo(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        image_path = op.join(work_dir, 'images.avi')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        writer = DatasetWriter(out_db_file,
                               image_path=image_path,
                               media='video')
        writer.addImage({'image': image})
        writer.addImage({'image': image})
        writer.close()

        assert op.exists(image_path)
        imreader = imageio.get_reader(image_path)
        num_frames = imreader.count_frames()
        imreader.close()
        assert num_frames == 2

    def test_record_imagefile_and_masks(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        mask_path = op.join(work_dir, 'masks')
        mask = np.zeros((100, 100), dtype=np.uint8)

        # Write existing images.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = op.join(work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)
        imageio.imwrite(op.join(image_path, 'existing2.jpg'), image)

        writer = DatasetWriter(out_db_file,
                               mask_path=mask_path,
                               rootdir=work_dir)

        # Write one image.
        writer.addImage({'imagefile': 'images/existing1.jpg', 'mask': mask})
        assert op.exists(mask_path)
        masksInDir = glob(op.join(mask_path, '*.png'))
        assert len(masksInDir) == 1

        # Write another image.
        writer.addImage({'imagefile': 'images/existing2.jpg', 'mask': mask})
        assert op.exists(mask_path)
        masksInDir = glob(op.join(mask_path, '*.png'))
        assert len(masksInDir) == 2

        # There should be no images.
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        assert len(imagesInDir) == 2

    def test_record_images_and_masks(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        image_path = op.join(work_dir, 'images')
        mask_path = op.join(work_dir, 'masks')
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        writer = DatasetWriter(out_db_file,
                               rootdir=work_dir,
                               image_path=image_path,
                               mask_path=mask_path)

        writer.addImage({'image': image, 'mask': mask})
        writer.addImage({'image': image})
        assert op.exists(image_path)
        assert op.exists(mask_path)
        imagesInDir = glob(op.join(image_path, '*.jpg'))
        assert len(imagesInDir) == 2
        masksInDir = glob(op.join(mask_path, '*.png'))
        assert len(masksInDir) == 1
        writer.close()

        assert op.exists(out_db_file)
        conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = conn.cursor()
        c.execute('SELECT imagefile,maskfile FROM images')
        image_entries = c.fetchall()
        assert len(image_entries) == 2
        assert image_entries[0][0] == 'images/0.jpg'
        assert image_entries[1][0] == 'images/1.jpg'
        assert image_entries[0][1] == 'masks/0.png'
        assert image_entries[1][1] is None

    def test_record_imagefile_maskfile_and_fields(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        timestamp = '2019-07-01 11:42:43.001000'
        timestamp_start = datetime.now()

        # Write existing images.
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image_path = op.join(work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)
        imageio.imwrite(op.join(image_path, 'existing2.jpg'), image)

        # Write existing masks.
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask_path = op.join(work_dir, 'masks')
        os.makedirs(mask_path)
        imageio.imwrite(op.join(mask_path, 'existing1.png'), mask)

        writer = DatasetWriter(out_db_file, rootdir=work_dir)
        writer.addImage({
            'imagefile': 'images/existing1.jpg',
            'maskfile': 'masks/existing1.png'
        })
        writer.addImage({
            'imagefile':
            'images/existing2.jpg',
            'timestamp':
            datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f'),
            'name':
            'myname',
            'score':
            0.5
        })
        writer.close()

        assert op.exists(out_db_file)
        conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = conn.cursor()
        c.execute(
            'SELECT imagefile,maskfile,width,height,timestamp,name,score FROM images'
        )
        image_entries = c.fetchall()

        # Check the first entry.
        assert len(image_entries) == 2
        assert image_entries[0][0] == 'images/existing1.jpg'
        assert image_entries[0][1] == 'masks/existing1.png'
        assert image_entries[0][2] == 200
        assert image_entries[0][3] == 100
        timestamp_written = datetime.strptime(image_entries[0][4],
                                              '%Y-%m-%d %H:%M:%S.%f')
        # The generated timestamp should be a fracetion of a second after timestamp_start.
        assert timestamp_written - timestamp_start < timedelta(seconds=10)
        assert image_entries[0][5] is None
        assert image_entries[0][6] is None

        # Check the second entry.
        assert image_entries[1][0] == 'images/existing2.jpg'
        assert image_entries[1][1] is None
        assert image_entries[1][2] == 200
        assert image_entries[1][3] == 100
        assert image_entries[1][4] == timestamp
        assert image_entries[1][5] == 'myname'
        assert image_entries[1][6] == 0.5

    def test_record_imagefile_maskfile_from_video(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')
        video_path = op.join(work_dir, 'images.avi')
        # Write a dummy video file.
        with open(video_path, 'w'):
            pass

        writer = DatasetWriter(out_db_file, rootdir=work_dir, media='video')
        writer.addImage({
            'imagefile': 'images.avi/000001',
            'maskfile': 'images.avi/000001',
            'width': 100,
            'height': 100
        })

        with pytest.raises(TypeError):  # Bad "height" type.
            writer.addImage({
                'imagefile': 'images.avi/000001',
                'width': 100,
                'height': 100.1
            })
        with pytest.raises(TypeError):  # Bad "width" type.
            writer.addImage({
                'imagefile': 'images.avi/000001',
                'width': np.ones(shape=(1, ), dtype=int),
                'height': 100.1
            })
        with pytest.raises(KeyError):  # Need keys "height", "width".
            writer.addImage({'imagefile': 'images.avi/000001'})
        with pytest.raises(ValueError):
            writer.addImage({'imagefile': 'nonexistant.avi/000001'})
        with pytest.raises(ValueError):
            writer.addImage({
                'imagefile': 'images.avi/000001',
                'maskfile': 'nonexistant.avi/000001',
                'width': 100,
                'height': 100
            })

    def test_record_objects(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')

        writer = DatasetWriter(out_db_file)

        # Require imagefile.
        with pytest.raises(KeyError):
            objectid = writer.addObject({})

        objectid1 = writer.addObject({
            'imagefile': 'myimagefile1',
            'x1': 10,
            'y1': 20,
            'width': 30,
            'height': 40,
            'name': 'car',
            'score': 0.5
        })
        objectid2 = writer.addObject({
            'imagefile': 'myimagefile2',
        })
        writer.close()

        assert op.exists(out_db_file)
        conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = conn.cursor()
        c.execute('SELECT * FROM objects')
        object_entries = c.fetchall()
        assert len(object_entries) == 2
        assert objectid1 == 1
        assert objectField(object_entries[0], 'objectid') == 1
        assert objectField(object_entries[0], 'imagefile') == 'myimagefile1'
        assert objectField(object_entries[0], 'x1') == 10
        assert objectField(object_entries[0], 'y1') == 20
        assert objectField(object_entries[0], 'width') == 30
        assert objectField(object_entries[0], 'height') == 40
        assert objectField(object_entries[0], 'name') == 'car'
        assert objectField(object_entries[0], 'score') == 0.5
        assert objectid2 == 2
        assert objectField(object_entries[1], 'objectid') == 2
        assert objectField(object_entries[1], 'imagefile') == 'myimagefile2'

    def test_object_types(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')

        writer = DatasetWriter(out_db_file)

        with pytest.raises(TypeError):
            writer.addObject({'imagefile': 'myimagefile', 'x1': 0.5})
        with pytest.raises(TypeError):
            writer.addObject({
                'imagefile': 'myimagefile',
                'y1': np.zeros(shape=(1, ), dtype=int)
            })
        with pytest.raises(TypeError):
            writer.addObject({'imagefile': 'myimagefile', 'width': 'badtype'})
        with pytest.raises(TypeError):
            writer.addObject({'imagefile': 'myimagefile', 'height': 0.5})
        with pytest.raises(TypeError):
            writer.addObject({'imagefile': 'myimagefile', 'score': 1})

    def test_record_matches(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')

        writer = DatasetWriter(out_db_file)
        objectid1 = writer.addObject({'imagefile': 'myimagefile'})
        objectid2 = writer.addObject({'imagefile': 'myimagefile'})
        objectid3 = writer.addObject({'imagefile': 'myimagefile'})
        objectid4 = writer.addObject({'imagefile': 'myimagefile'})
        # Add a match.
        match1 = writer.addMatch(objectid=objectid1)
        writer.addMatch(objectid=objectid2, match=match1)
        writer.addMatch(objectid=objectid3, match=match1)
        # Add another match.
        match2 = writer.addMatch(objectid=objectid1)
        writer.addMatch(objectid=objectid4, match=match2)
        writer.close()

        assert op.exists(out_db_file)
        conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = conn.cursor()
        c.execute('SELECT objectid,match FROM matches')
        match_entries = c.fetchall()
        assert len(match_entries) == 5
        assert match_entries[0][0] == objectid1
        assert match_entries[1][0] == objectid2
        assert match_entries[2][0] == objectid3
        assert match_entries[3][0] == objectid1
        assert match_entries[4][0] == objectid4
        assert match_entries[0][1] == match1
        assert match_entries[1][1] == match1
        assert match_entries[2][1] == match1
        assert match_entries[3][1] == match2
        assert match_entries[4][1] == match2

    def test_context(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')

        # Write existing images.
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = op.join(work_dir, 'images')
        os.makedirs(image_path)
        imageio.imwrite(op.join(image_path, 'existing1.jpg'), image)

        with DatasetWriter(out_db_file, rootdir=work_dir) as writer:
            writer.addImage({'imagefile': 'images/existing1.jpg'})

        assert op.exists(out_db_file)
        conn = sqlite3.connect('file:%s?mode=ro' % out_db_file, uri=True)
        c = conn.cursor()
        c.execute('SELECT imagefile FROM images')
        imagefiles = c.fetchall()
        assert len(imagefiles) == 1
        assert imagefiles[0][0] == 'images/existing1.jpg'

    def test_failed_record_imagefile(self, work_dir):
        out_db_file = op.join(work_dir, 'out.db')

        writer = DatasetWriter(out_db_file)

        # Try to record an imagefile of an image that does not exist.
        with pytest.raises(
                ValueError):  # Cant use FileExistsError in Python 2.
            writer.addImage({'imagefile': 'nonexisting.jpg'})

    def test_failed_record_maskfile(self, work_dir):
        ''' Should fail to record an maskfile of mask that does not exist. '''
        out_db_file = op.join(work_dir, 'out.db')
        image_path = op.join(work_dir, 'images')
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        writer = DatasetWriter(out_db_file, image_path=image_path)

        # Cant use FileExistsError in Python 2.
        with pytest.raises(ValueError):
            writer.addImage({'image': image, 'maskfile': 'nonexisting.png'})

    def test_overwrite_db(self, work_dir):
        ''' Test "overwrite" for the .db file. '''
        out_db_file = op.join(work_dir, 'out.db')

        writer = DatasetWriter(out_db_file)
        writer = DatasetWriter(out_db_file)  # Should open to change.
        writer = DatasetWriter(out_db_file, overwrite=True)  # Overwrites.

    def test_overwrite_video(self, work_dir):
        out_db_file1 = op.join(work_dir, 'out1.db')
        out_db_file2 = op.join(work_dir, 'out2.db')
        out_db_file3 = op.join(work_dir, 'out3.db')
        out_video_path = op.join(work_dir, 'images.avi')

        # Check "overwrite" for the video file.
        writer = DatasetWriter(out_db_file1,
                               image_path=out_video_path,
                               media='video')
        writer.addImage({'image': np.zeros((100, 100, 3), dtype=np.uint8)})
        writer.close()
        with pytest.raises(
                ValueError):  # Cant use FileExistsError in Python 2.
            writer = DatasetWriter(out_db_file2,
                                   image_path=out_video_path,
                                   media='video')
        writer = DatasetWriter(out_db_file3,
                               image_path=out_video_path,
                               media='video',
                               overwrite=True)
        writer.close()

    def test_overwrite_images(self, work_dir):
        out_db_file1 = op.join(work_dir, 'out1.db')
        out_db_file2 = op.join(work_dir, 'out2.db')
        out_db_file3 = op.join(work_dir, 'out3.db')
        out_image_path = op.join(work_dir, 'images')

        # Check "overwrite" for the video file.
        DatasetWriter(out_db_file1, image_path=out_image_path)
        with pytest.raises(
                ValueError):  # Cant use FileExistsError in Python 2.
            DatasetWriter(out_db_file2, image_path=out_image_path)
        DatasetWriter(out_db_file3, image_path=out_image_path, overwrite=True)
