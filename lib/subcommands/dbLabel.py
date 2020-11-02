import os, sys, os.path as op
import numpy as np
import logging
import cv2
from scipy.cluster import hierarchy

from lib.backend import backendDb
from lib.backend import backendMedia
from lib.subcommands import dbGui


def add_parsers(subparsers):
    labelAzimuthParser(subparsers)


def labelAzimuthParser(subparsers):
    parser = subparsers.add_parser(
        'labelAzimuth',
        description='''Go through objects and label yaw (azimuth)
    by either accepting one of the close yaw values from a map,
    or by assigning a value manually.''')
    parser.set_defaults(func=labelAzimuth)
    parser.add_argument('--winsize', type=int, default=500)
    parser.add_argument('--scenes_dir',
                        required=True,
                        help='Directory with scenes by camera.')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--where_object',
                        default='TRUE',
                        help='the SQL "where" clause for the "objects" table.')
    parser.add_argument('--key_dict',
                        default='{"-": "previous", "=": "next", 27: "exit"}')


def labelAzimuth(c, args):
    from lib.backendScenes import Window
    from scenes.lib.cvScrollZoomWindow import Window
    from scenes.lib.homography import getFrameFlattening, getHfromPose
    from scenes.lib.cache import PoseCache, MapsCache
    from scenes.lib.warp import transformPoint

    os.environ['SCENES_PATH'] = args.scenes_dir

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)
    key_reader = dbGui.KeyReader(args.key_dict)

    c.execute('SELECT * FROM objects WHERE (%s)' % args.where_object)
    object_entries = c.fetchall()
    logging.info('Found %d objects in db.' % len(object_entries))
    if len(object_entries) == 0:
        return

    if args.shuffle:
        np.random.shuffle(object_entries)

    # Cached poses and azimuth maps.
    topdown_azimuths = MapsCache('topdown_azimuth')
    poses = PoseCache()

    button = 0
    index_object = 0
    another_object = True
    char_list = []
    while button != 27:
        go_next_object = False
        update_yaw_in_db = False

        if another_object:
            another_object = False

            logging.info(' ')
            logging.info('Object %d out of %d' %
                         (index_object, len(object_entries)))
            object_entry = object_entries[index_object]
            objectid = backendDb.objectField(object_entry, 'objectid')
            bbox = backendDb.objectField(object_entry, 'bbox')
            roi = backendDb.objectField(object_entry, 'roi')
            imagefile = backendDb.objectField(object_entry, 'imagefile')
            # Update yaw inside the loop in case it was just assigned.
            c.execute(
                'SELECT value FROM properties WHERE objectid=? AND key="yaw"',
                (objectid, ))
            yaw = c.fetchone()
            yaw = float(yaw[0]) if yaw is not None else None

            y, x = roi[0] * 0.3 + roi[2] * 0.7, roi[1] * 0.5 + roi[3] * 0.5

            try:
                flattening = _getFlatteningFromImagefile(
                    poses, imagefile, y, x)
            except:
                # A hack that allowed me write all the images into one dir but
                # keep info about imagefile.
                c.execute('SELECT name FROM images WHERE imagefile=?',
                          (imagefile, ))
                image_name = c.fetchone()[0]
                flattening = _getFlatteningFromImagefile(
                    poses, image_name, y, x)

            axis_x = np.linalg.norm(np.asarray(bbox[2:4]), ord=2)
            axis_y = axis_x * flattening

            display = imreader.imread(imagefile)
            window = AzimuthWindow(display,
                                   x,
                                   y,
                                   axis_x,
                                   axis_y,
                                   winsize=args.winsize)
            if yaw is not None:
                logging.info('Yaw is: %.0f' % yaw)
                window.yaw = yaw
            window.update_cached_zoomed_img()
            window.redraw()

        button = cv2.waitKey(50)
        action = key_reader.parse(button) if button is not -1 else None

        if action == 'delete':
            c.execute('DELETE FROM properties WHERE objectid=? AND key="yaw"',
                      (objectid, ))
            logging.info('Yaw is deleted.')
            go_next_object = True
            char_list = []

        # Entry in keyboard.
        if button >= ord('0') and button <= ord('9') or button == ord('.'):
            char_list += chr(button)
            logging.debug('Added %s character to number and got %s' %
                          (chr(button), ''.join(char_list)))
        # Enter accepts GUI or keyboard entry.
        elif button == 13:
            if char_list:  # After keyboard entry.
                number_str = ''.join(char_list)
                char_list = []
                try:
                    logging.info('Accepting entry from the keyboard.')
                    yaw = float(number_str)
                    update_yaw_in_db = True
                    go_next_object = True
                except ValueError:
                    logging.warning('Could not convert entered %s to number.' %
                                    number_str)
                    continue
            else:  # Just navigation.
                logging.info('A navigation Enter.')
                go_next_object = True
        # Entry in GUI.
        elif window.selected == True:
            logging.info('Accepting entry from GUI.')
            yaw = window.yaw
            update_yaw_in_db = True
            go_next_object = True
        # No entry:
        else:
            yaw = None

        # Entry happened one way or the other.
        # Update the yaw and go to the next object.
        if update_yaw_in_db:
            c.execute(
                'SELECT COUNT(1) FROM properties WHERE objectid=? AND key="yaw"',
                (objectid, ))
            if c.fetchone()[0] == 0:
                c.execute(
                    'INSERT INTO properties(value,key,objectid) VALUES (?,"yaw",?)',
                    (str(yaw), objectid))
            else:
                c.execute(
                    'UPDATE properties SET value=? WHERE objectid=? AND key="yaw"',
                    (str(yaw), objectid))
            logging.info('Yaw is assigned to %.f' % yaw)

        # Navigation.
        if action == 'previous':
            logging.debug('previous object')
            if index_object > 0:
                index_object -= 1
                another_object = True
            else:
                logging.warning('Already at the first object.')
        elif action == 'next' or go_next_object == True:
            logging.debug('next object')
            if index_object < len(object_entries) - 1:
                index_object += 1
                another_object = True
            else:
                logging.warning(
                    'Already at the last object. Press Esc to save and exit.')
