import os, sys, os.path as op
import numpy as np
import cv2
import logging
from ast import literal_eval
from pprint import pformat
import imageio

from lib.utils import util
from lib.backend import backendDb
from lib.backend import backendMedia


def add_parsers(subparsers):
    examineImagesParser(subparsers)
    examineObjectsParser(subparsers)
    labelObjectsParser(subparsers)
    examineMatchesParser(subparsers)
    labelMatchesParser(subparsers)


class KeyReader:
    '''
    A mapper from keyboard buttons to actions.
    '''
    def __init__(self, keysmap_str):
        '''
        Args:
          key_dict: a string that can be parsed as dict Key->Action, where:
                    Key - a string of one char or a an int for ASCII
                    Action - a string.
                    Actions 'exit', 'previous', and 'next' must be specified.
        Returns:
          Parsed dict.
        '''
        logging.debug('Got keys map string as: %s' % keysmap_str)
        keysmap = literal_eval(keysmap_str)
        logging.info('Keys map was parsed as: %s' % pformat(keysmap))
        for value in ['exit', 'previous', 'next']:
            if not value in keysmap.values():
                raise ValueError('"%s" must be in the values of keysmap, '
                                 'now the values are: %s' %
                                 (value, keysmap.values()))
        for key in keysmap:
            if not isinstance(key, int) and not (isinstance(key, str)
                                                 and len(key) == 1):
                raise ValueError(
                    'Each key in key_dict must be an integer (for ASCII code) '
                    'or a string of a single character. Got key: %s' % key)
        self.keysmap = keysmap

    def parse(self, button):
        ''' Get the corresponding action for a pressed button. '''

        if button == -1:  # cv2 code for no press.
            #logging.debug('No key was pressed')
            return None

        if chr(button) in self.keysmap:
            # Get the character from ASCII, since character is in the keymap.
            logging.info('Found char "%s" for pressed ASCII %d in the table.' %
                         (chr(button), button))
            button = chr(button)

        if button in self.keysmap:
            logging.info('Value for pressed "%s" is "%s".' %
                         (str(button), self.keysmap[button]))
            return self.keysmap[button]
        else:
            logging.info('No value for pressed "%s".' % str(button))
            return None


def examineImagesParser(subparsers):
    parser = subparsers.add_parser(
        'examineImages',
        description='Loop through images. Possibly, assign names to images.')
    parser.set_defaults(func=examineImages)
    parser.add_argument('--where_image',
                        default='TRUE',
                        help='the SQL "where" clause for the "images" table.')
    parser.add_argument('--mask_mapping_dict',
                        help='A mapping to display values in maskfile. '
                        'E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask_aside',
                       action='store_true',
                       help='Image and mask side by side.')
    group.add_argument('--mask_alpha',
                       type=float,
                       help='Mask will be overlayed on the image.'
                       'Transparency to overlay the label mask with, '
                       '1 means cant see the image behind the mask.')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--with_objects',
                        action='store_true',
                        help='draw all objects on top of the image.')
    parser.add_argument('--with_score',
                        action='store_true',
                        help='draw image score on top of the image.')
    parser.add_argument('--with_imagefile',
                        action='store_true',
                        help='draw imagefile on top of the image.')
    parser.add_argument('--winsize', type=int, default=1000)
    parser.add_argument(
        '--key_dict',
        default=
        '{"-": "previous", "=": "next", " ": "snapshot", 127: "delete", 27: "exit"}'
    )
    parser.add_argument(
        '--snapshot_dir',
        help='If provided, snapshots will be written to "snapshot_dir".')


def examineImages(c, args):
    cv2.namedWindow("examineImages")

    c.execute('SELECT * FROM images WHERE (%s)' % args.where_image)
    image_entries = c.fetchall()
    logging.info('%d images found.' % len(image_entries))
    if len(image_entries) == 0:
        logging.error('There are no images. Exiting.')
        return

    if args.shuffle:
        np.random.shuffle(image_entries)

    if args.snapshot_dir and not op.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For parsing keys.
    key_reader = KeyReader(args.key_dict)

    # For overlaying masks.
    labelmap = literal_eval(
        args.mask_mapping_dict) if args.mask_mapping_dict else None
    logging.info('Parsed mask_mapping_dict to %s' % pformat(labelmap))

    index_image = 0

    while True:  # Until a user hits the key for the "exit" action.

        image_entry = image_entries[index_image]
        imagefile = backendDb.imageField(image_entry, 'imagefile')
        maskfile = backendDb.imageField(image_entry, 'maskfile')
        imname = backendDb.imageField(image_entry, 'name')
        imscore = backendDb.imageField(image_entry, 'score')
        logging.info('Imagefile "%s"' % imagefile)
        logging.debug('Image name="%s", score=%s' % (imname, imscore))
        image = imreader.imread(imagefile)

        # Overlay the mask.
        if maskfile is not None:
            mask = imreader.maskread(maskfile)
            if args.mask_aside:
                image = util.drawMaskAside(image, mask, labelmap=labelmap)
            elif args.mask_alpha is not None:
                image = util.drawMaskOnImage(image,
                                             mask,
                                             alpha=args.mask_alpha,
                                             labelmap=labelmap)
        else:
            logging.info('No mask for this image.')

        # Put the objects on top of the image.
        if args.with_objects:
            c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
            object_entries = c.fetchall()
            logging.info('Found %d objects for image %s' %
                         (len(object_entries), imagefile))
            for object_entry in object_entries:
                objectid = backendDb.objectField(object_entry, 'objectid')
                roi = backendDb.objectField(object_entry, 'roi')
                score = backendDb.objectField(object_entry, 'score')
                name = backendDb.objectField(object_entry, 'name')
                logging.info('objectid: %d, roi: %s, score: %s, name: %s' %
                             (objectid, roi, score, name))
                c.execute('SELECT * FROM polygons WHERE objectid=?',
                          (objectid, ))
                polygon_entries = c.fetchall()
                if len(polygon_entries) > 0:
                    logging.info('showing object with a polygon.')
                    polygon = [(int(backendDb.polygonField(p, 'x')),
                                int(backendDb.polygonField(p, 'y')))
                               for p in polygon_entries]
                    util.drawScoredPolygon(image,
                                           polygon,
                                           label=name,
                                           score=score)
                elif roi is not None:
                    logging.info('showing object with a bounding box.')
                    util.drawScoredRoi(image, roi, label=name, score=score)
                else:
                    logging.warning(
                        'Neither polygon, nor bbox is available for objectid %d'
                        % objectid)

        # Display an image, wait for the key from user, and parse that key.
        scale = float(args.winsize) / max(list(image.shape[0:2]))
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        # Overlay imagefile.
        if args.with_imagefile:
            util.drawTextOnImage(
                scaled_image,
                op.basename(backendMedia.normalizeSeparators(imagefile)))
        # Overlay score.
        # TODO: add y offset, if necessary
        if args.with_score:
            util.drawTextOnImage(scaled_image, '%.3f' % imscore)
        # Display
        cv2.imshow('examineImages', scaled_image[:, :, ::-1])
        action = key_reader.parse(cv2.waitKey(-1))
        if action is None:
            # User pressed something which does not have an action.
            continue
        elif action == 'snapshot':
            if args.snapshot_dir:
                snaphot_path = op.join(args.snapshot_dir,
                                       '%08d.png' % index_image)
                logging.info('Making a snapshot at path: %s' % snaphot_path)
                imageio.imwrite(snaphot_path, image)
            else:
                logging.warning(
                    'The user pressed a snapshot key, but snapshot_dir is not '
                    'specified. Will not write a snapshot.')
        elif action == 'delete':
            backendDb.deleteImage(c, imagefile)
            del image_entries[index_image]
            if len(image_entries) == 0:
                logging.warning('Deleted the last image.')
                break
            index_image += 1
        elif action == 'exit':
            break
        elif action == 'previous':
            index_image -= 1
        elif action == 'next':
            index_image += 1
        else:
            # User pressed something else which has an assigned action, assume it is a new name.
            logging.info('Setting name "%s" to imagefile "%s"' %
                         (action, imagefile))
            c.execute('UPDATE images SET name=? WHERE imagefile=?' (action,
                                                                    imagefile))
        index_image = index_image % len(image_entries)

    cv2.destroyWindow("examineImages")


def examineObjectsParser(subparsers):
    parser = subparsers.add_parser('examineObjects',
                                   description='Loop through objects.')
    parser.set_defaults(func=examineObjects)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--where_object',
                        default='TRUE',
                        help='the SQL "where" clause for the "objects" table.')
    parser.add_argument('--winsize', type=int, default=1000)
    parser.add_argument(
        '--key_dict',
        default=
        '{"-": "previous", "=": "next", 27: "exit", 127: "delete", " ": "unname"}'
    )
    # TODO: add display of mask.


def examineObjects(c, args):
    cv2.namedWindow("examineObjects")

    c.execute('SELECT COUNT(*) FROM objects WHERE (%s) ' % args.where_object)
    logging.info('Found %d objects in db.' % c.fetchone()[0])

    c.execute('SELECT DISTINCT imagefile FROM objects WHERE (%s) ' %
              args.where_object)
    image_entries = c.fetchall()
    logging.info('%d images found.' % len(image_entries))
    if len(image_entries) == 0:
        logging.error('There are no images. Exiting.')
        return

    if args.shuffle:
        np.random.shuffle(image_entries)

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For parsing keys.
    key_reader = KeyReader(args.key_dict)

    index_image = 0
    index_object = 0

    # Iterating over images, because for each image we want to show all objects.
    while True:  # Until a user hits the key for the "exit" action.

        (imagefile, ) = image_entries[index_image]
        logging.info('Imagefile "%s"' % imagefile)
        image = imreader.imread(imagefile)
        scale = float(args.winsize) / max(image.shape[0:2])
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        c.execute(
            'SELECT * FROM objects WHERE imagefile=? AND (%s)' %
            args.where_object, (imagefile, ))
        object_entries = c.fetchall()
        logging.info('Found %d objects for image %s' %
                     (len(object_entries), imagefile))

        # Put the objects on top of the image.
        if len(object_entries) > 0:
            assert index_object < len(object_entries)
            object_entry = object_entries[index_object]
            objectid = backendDb.objectField(object_entry, 'objectid')
            roi = backendDb.objectField(object_entry, 'roi')
            score = backendDb.objectField(object_entry, 'score')
            name = backendDb.objectField(object_entry, 'name')
            scaledroi = [int(scale * r)
                         for r in roi]  # For displaying the scaled image.
            logging.info('objectid: %d, roi: %s, score: %s, name: %s' %
                         (objectid, roi, score, name))
            c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid, ))
            polygon_entries = c.fetchall()
            if len(polygon_entries) > 0:
                logging.info('showing object with a polygon.')
                polygon = [(backendDb.polygonField(p, 'x'),
                            backendDb.polygonField(p, 'y'))
                           for p in polygon_entries]
                logging.debug('nonscaled polygon: %s' % pformat(polygon))
                polygon = [(int(scale * x), int(scale * y))
                           for x, y in polygon]
                logging.debug('scaled polygon: %s' % pformat(polygon))
                util.drawScoredPolygon(image, polygon, label=None, score=score)
            elif roi is not None:
                logging.info('showing object with a bounding box.')
                util.drawScoredRoi(image, scaledroi, label=None, score=score)
            else:
                raise Exception(
                    'Neither polygon, nor bbox is available for objectid %d' %
                    objectid)
            c.execute('SELECT key,value FROM properties WHERE objectid=?',
                      (objectid, ))
            properties = c.fetchall()
            if name is not None:
                properties.append(('name', name))
            if score is not None:
                properties.append(('score', score))
            for iproperty, (key, value) in enumerate(properties):
                cv2.putText(image, '%s: %s' % (key, value),
                            (scaledroi[3] + 10,
                             scaledroi[0] - 10 + util.SCALE * (iproperty + 1)),
                            util.FONT, util.FONT_SIZE, (0, 0, 0),
                            util.THICKNESS)
                cv2.putText(image, '%s: %s' % (key, value),
                            (scaledroi[3] + 10,
                             scaledroi[0] - 10 + util.SCALE * (iproperty + 1)),
                            util.FONT, util.FONT_SIZE, (255, 255, 255),
                            util.THICKNESS - 1)
                logging.info('objectid: %d. %s = %s.' % (objectid, key, value))

        # Display an image, wait for the key from user, and parse that key.
        cv2.imshow('examineObjects', image[:, :, ::-1])
        action = key_reader.parse(cv2.waitKey(-1))
        if action is None:
            # User pressed something which does not have an action.
            continue
        elif action == 'exit':
            break
        elif action == 'previous':
            index_object -= 1
            if index_object < 0:
                index_image -= 1
                index_object = 0
        elif action == 'next':
            index_object += 1
            if index_object >= len(object_entries):
                index_image += 1
                index_object = 0
        elif action == 'delete' and len(object_entries) > 0:
            backendDb.deleteObject(c, objectid)
            del object_entries[index_object]
            if index_object >= len(object_entries):
                index_image += 1
                index_object = 0
        elif action == 'unname' and len(object_entries) > 0:
            logging.info('Remove the name from objectid "%s"' % objectid)
            c.execute('UPDATE objects SET name=NULL WHERE objectid=?',
                      (objectid, ))
        index_image = index_image % len(image_entries)

    cv2.destroyWindow("examineObjects")


def labelObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'labelObjects',
        description='Loop through objects and manually label them, '
        'i.e. assign the value of user-defined property.')
    parser.set_defaults(func=labelObjects)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--where_object',
                        default='TRUE',
                        help='the SQL "where" clause for the "objects" table.')
    parser.add_argument('--winsize', type=int, default=1000)
    parser.add_argument('--property',
                        required=True,
                        help='name of the property being labelled')
    parser.add_argument(
        '--key_dict',
        default=
        '{"-": "previous", "=": "next", 27: "exit", 127: "delete_label", '
        ' "r": "red", "g": "green", "b": "blue"}')


def labelObjects(c, args):
    cv2.namedWindow("labelObjects")

    c.execute('SELECT COUNT(*) FROM objects WHERE (%s) ' % args.where_object)
    logging.info('Found %d objects in db.' % c.fetchone()[0])

    c.execute('SELECT * FROM objects WHERE (%s)' % args.where_object)
    object_entries = c.fetchall()
    logging.info('Found %d objects in db.' % len(object_entries))
    if len(object_entries) == 0:
        return

    if args.shuffle:
        np.random.shuffle(object_entries)

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For parsing keys.
    key_reader = KeyReader(args.key_dict)

    button = 0
    index_object = 0
    another_object = True
    while button != 27:
        go_next_object = False

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
            logging.info('imagefile: %s' % imagefile)
            image = imreader.imread(imagefile)

            # Display an image, wait for the key from user, and parse that key.
            scale = float(args.winsize) / max(image.shape[0:2])
            logging.debug('Will resize image and annotations with scale: %f' %
                          scale)
            image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

            logging.info('objectid: %d, roi: %s' % (objectid, roi))
            c.execute('SELECT * FROM polygons WHERE objectid=?', (objectid, ))
            polygon_entries = c.fetchall()
            if len(polygon_entries) > 0:
                logging.info('showing object with a polygon.')
                polygon = [(backendDb.polygonField(p, 'x'),
                            backendDb.polygonField(p, 'y'))
                           for p in polygon_entries]
                logging.debug('nonscaled polygon: %s' % pformat(polygon))
                polygon = [(int(scale * p[0]), int(scale * p[1]))
                           for p in polygon]
                logging.debug('scaled polygon: %s' % pformat(polygon))
                util.drawScoredPolygon(image, polygon, label=None, score=None)
            elif roi is not None:
                logging.info('showing object with a bounding box.')
                logging.debug('nonscaled roi: %s' % pformat(roi))
                roi = [int(scale * r)
                       for r in roi]  # For displaying the scaled image.
                logging.debug('scaled roi: %s' % pformat(roi))
                util.drawScoredRoi(image, roi, label=None, score=None)
            else:
                raise Exception(
                    'Neither polygon, nor bbox is available for objectid %d' %
                    objectid)
            c.execute(
                'SELECT key,value FROM properties WHERE objectid=? AND key=?',
                (objectid, args.property))
            # TODO: Multiple properties are possible because there is no
            #       contraint on uniqueness on table properties(objectid,key).
            #       Change when the uniqueness constraint is added to the
            #       database schema. On the other hand, it's a feature.
            properties = c.fetchall()
            if len(properties) > 1:
                logging.warning(
                    'Multiple values for object %s and property %s. '
                    'If reassigned, both will be changed' %
                    (objectid, args.property))

            for iproperty, (key, value) in enumerate(properties):
                cv2.putText(image, '%s: %s' % (key, value),
                            (10, util.SCALE * (iproperty + 1)), util.FONT,
                            util.FONT_SIZE, (0, 0, 0), util.THICKNESS)
                cv2.putText(image, '%s: %s' % (key, value),
                            (10, util.SCALE * (iproperty + 1)), util.FONT,
                            util.FONT_SIZE, (255, 255, 255),
                            util.THICKNESS - 1)
                logging.info('objectid: %d. %s = %s.' % (objectid, key, value))

        cv2.imshow('labelObjects', image[:, :, ::-1])
        action = key_reader.parse(cv2.waitKey(-1))
        if action == 'exit':
            break
        elif action == 'delete_label' and any_object_in_focus:
            logging.info('Remove label from objectid "%s"' % objectid)
            c.execute('DELETE FROM properties WHERE objectid=? AND key=?',
                      (objectid, args.property))
            go_next_object = True
        elif action is not None and action not in ['previous', 'next']:
            # User pressed something else which has an assigned action,
            # assume it is a new value.
            logging.info('Setting label "%s" to objectid "%s"' %
                         (action, objectid))
            if len(properties) > 0:
                c.execute('DELETE FROM properties WHERE objectid=? AND key=?',
                          (objectid, args.property))
            c.execute(
                'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                (objectid, args.property, str(action)))
            go_next_object = True
        # Navigation.
        if action == 'previous':
            logging.debug('previous object')
            another_object = True
            if index_object > 0:
                index_object -= 1
            else:
                logging.warning('Already at the first object.')
        elif action == 'next' or go_next_object == True:
            logging.debug('next object')
            another_object = True
            if index_object < len(object_entries) - 1:
                index_object += 1
            else:
                logging.warning(
                    'Already at the last object. Press Esc to save and exit.')

    cv2.destroyWindow("labelObjects")


def examineMatchesParser(subparsers):
    parser = subparsers.add_parser(
        'examineMatches',
        description=
        '''Browse through database and see car bboxes on top of images.
                   Any key will scroll to the next image.''')
    parser.set_defaults(func=examineMatches)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--winsize', type=int, default=1000)
    parser.add_argument(
        '--key_dict',
        default='{"-": "previous", "=": "next", " ": "next", 27: "exit"}')


def examineMatches(c, args):
    cv2.namedWindow("examineMatches")

    c.execute('SELECT DISTINCT(match) FROM matches')
    match_entries = c.fetchall()

    if args.shuffle:
        np.random.shuffle(match_entries)

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For parsing keys.
    key_reader = KeyReader(args.key_dict)

    index_match = 0

    # Iterating over images, because for each image we want to show all objects.
    while True:  # Until a user hits the key for the "exit" action.

        (match, ) = match_entries[index_match]
        c.execute(
            'SELECT * FROM objects WHERE objectid IN '
            '(SELECT objectid FROM matches WHERE match=?)', (match, ))
        object_entries = c.fetchall()
        logging.info('Found %d objects for match %d' %
                     (len(object_entries), match))

        images = []
        for object_entry in object_entries:
            imagefile = backendDb.objectField(object_entry, 'imagefile')
            objectid = backendDb.objectField(object_entry, 'objectid')
            roi = backendDb.objectField(object_entry, 'roi')
            score = backendDb.objectField(object_entry, 'score')

            image = imreader.imread(imagefile)
            util.drawScoredRoi(image, roi, score=score)

            scale = float(args.winsize) / max(image.shape[0:2])
            image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
            images.append(image)

        # Assume all images have the same size for now.
        # Assume there are not so many matches.
        image = np.hstack(images)

        # Display an image, wait for the key from user, and parse that key.
        cv2.imshow('examineMatches', image[:, :, ::-1])
        action = key_reader.parse(cv2.waitKey(-1))
        if action is None:
            # User pressed something which does not have an action.
            continue
        elif action == 'exit':
            break
        elif action == 'previous':
            index_match -= 1
        elif action == 'next':
            index_match += 1
        else:
            # User pressed something else which has an assigned action,
            # assume it is a new name.
            logging.info('Setting name "%s" to imagefile "%s"' %
                         (action, imagefile))
            c.execute('UPDATE images SET name=? WHERE imagefile=?' (action,
                                                                    imagefile))
        index_match = index_match % len(match_entries)

    cv2.destroyWindow("examineMatches")


# Helper global vars for mouse callback _monitorPressRelease.
xpress, ypress = None, None
mousePressed = False
mouseReleased = False


def _monitorPressRelease(event, x, y, flags, param):
    ''' A mouse callback for labelling matches. '''

    global xpress, ypress, mousePressed, mouseReleased

    if event == cv2.EVENT_LBUTTONDOWN:
        logging.debug('Left mouse button pressed at %d,%d' % (x, y))
        xpress, ypress = x, y
        assert not mouseReleased
        mousePressed = True

    elif event == cv2.EVENT_LBUTTONUP:
        xpress, ypress = x, y
        logging.debug('Left mouse button released at %d,%d' % (x, y))
        assert not mousePressed
        mouseReleased = True


def _drawMatch(img, roi1, roi2, yoffset):
    def _getCenter(roi):
        return int(0.5 * roi[1] + 0.5 * roi[3]), int(0.25 * roi[0] +
                                                     0.75 * roi[2])

    roi2[0] += yoffset
    roi2[2] += yoffset
    util.drawScoredRoi(img, roi1)
    util.drawScoredRoi(img, roi2)
    center1 = _getCenter(roi1)
    center2 = _getCenter(roi2)
    cv2.line(img, center1, center2, [0, 0, 255], thickness=2)


def _findPressedObject(x, y, cars):
    for i in range(len(cars)):
        roi = backendDb.objectField(cars[i], 'roi')
        if x >= roi[1] and x < roi[3] and y >= roi[0] and y < roi[2]:
            return i
    return None


def labelMatchesParser(subparsers):
    parser = subparsers.add_parser(
        'labelMatches',
        description=
        'Loop through sequential image pairs and label matching objects '
        'on the two images of each pair. '
        'At the start you should see a window with two images one under another. '
        'Press and hold the mouse at a Bbox in the top, and release at a Bbox in the bottom. '
        'That will add a match between this pair of Bboxes. '
        'If one of the two boxes were matched to something already, match will not be added. '
        'Clicking on a box in the top image, and press DEL '
        'will remove a match if the top box was matched. '
        'Pressing "prev" (key "-" by default) takes you to the previous image. '
        'Pressing "next" (key "=" by default) takes you to the next image. '
        'Press "exit" (key "Esc" by default) to save changes and exit. '
        'Pass "min_imagefile" to start with a certain image pair.')
    parser.set_defaults(func=labelMatches)
    parser.add_argument('--where_image',
                        default='TRUE',
                        help='the SQL "where" clause for the "images" table.')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--winsize', type=int, default=1000)
    parser.add_argument(
        '--key_dict',
        default=
        '{"-": "previous", "=": "next", 127: "delete_match", 27: "exit"}')


def labelMatches(c, args):
    cv2.namedWindow("labelMatches")
    cv2.setMouseCallback('labelMatches', _monitorPressRelease)
    global mousePressed, mouseReleased, xpress, ypress

    c.execute('SELECT imagefile FROM images WHERE %s' % args.where_image)
    image_entries = c.fetchall()
    logging.debug('Found %d images' % len(image_entries))
    if len(image_entries) < 2:
        logging.error('Found only %d images. Quit.' % len(image_entries))
        return

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For parsing keys.
    key_reader = KeyReader(args.key_dict)

    index_image = 1
    action = None
    while action != 'exit' and index_image < len(image_entries):
        (imagefile1, ) = image_entries[index_image - 1]
        (imagefile2, ) = image_entries[index_image]

        img1 = imreader.imread(imagefile1)
        img2 = imreader.imread(imagefile2)
        # offset of the 2nd image, when they are stacked
        yoffset = img1.shape[0]

        # Make sure images have the same width, so that we can stack them.
        # That will happen when the video changes, so we'll skip that pair.
        if img1.shape[1] != img2.shape[1]:
            logging.warning('Skipping image pair "%s" and "%s" '
                            'since they are of different width.' %
                            (imagefile1, imagefile2))
            index_image += 1

        # get objects from both images
        c.execute('SELECT * FROM objects WHERE imagefile=? ', (imagefile1, ))
        objects1 = c.fetchall()
        logging.info('%d objects found for %s' % (len(objects1), imagefile1))
        c.execute('SELECT * FROM objects WHERE imagefile=? ', (imagefile2, ))
        objects2 = c.fetchall()
        logging.info('%d objects found for %s' % (len(objects2), imagefile2))

        # draw cars in both images
        for object_ in objects1:
            util.drawScoredRoi(img1, backendDb.objectField(object_, 'roi'))
        for object_ in objects2:
            util.drawScoredRoi(img2, backendDb.objectField(object_, 'roi'))

        i1 = i2 = None  # Matches selected with a mouse.

        selectedMatch = None
        needRedraw = True
        action = None
        # Stay in side the loop until a key is pressed.
        while action is None:

            img_stack = np.vstack((img1, img2))

            if needRedraw:

                # find existing matches, and make a map
                matchesOf1 = {}
                matchesOf2 = {}
                for j1 in range(len(objects1)):
                    object1 = objects1[j1]
                    for j2 in range(len(objects2)):
                        object2 = objects2[j2]
                        c.execute(
                            'SELECT match FROM matches WHERE objectid = ? INTERSECT '
                            'SELECT match FROM matches WHERE objectid = ?',
                            (backendDb.objectField(object1, 'objectid'),
                             backendDb.objectField(object2, 'objectid')))
                        matches = c.fetchall()
                        if len(matches) > 0:
                            assert len(matches) == 1  # No duplicate matches.
                            roi1 = backendDb.objectField(object1, 'roi')
                            roi2 = backendDb.objectField(object2, 'roi')
                            _drawMatch(img_stack, roi1, roi2, yoffset)
                            matchesOf1[j1] = matches[0][0]
                            matchesOf2[j2] = matches[0][0]

                # draw image
                scale = float(args.winsize) / max(img_stack.shape[0:2])
                img_show = cv2.resize(img_stack,
                                      dsize=(0, 0),
                                      fx=scale,
                                      fy=scale)
                cv2.imshow('labelMatches', img_show[:, :, ::-1])
                logging.info('Will draw %d matches found between the pair' %
                             len(matchesOf1))
                needRedraw = False

            # process mouse callback effect (button has been pressed)
            if mousePressed:
                i2 = None  # reset after the last unsuccessful match
                logging.debug('Pressed  x=%d, y=%d' % (xpress, ypress))
                xpress /= scale
                ypress /= scale
                i1 = _findPressedObject(xpress, ypress, objects1)
                if i1 is not None:
                    logging.debug('Found pressed object: %d' % i1)
                mousePressed = False

            # process mouse callback effect (button has been released)
            if mouseReleased:
                logging.debug('released x=%d, y=%d' % (xpress, ypress))
                xpress /= scale
                ypress /= scale
                i2 = _findPressedObject(xpress, ypress - yoffset, objects2)
                if i2 is not None:
                    logging.debug('Found released object: %d' % i2)
                mouseReleased = False

            # If we could find pressed and released objects, add match
            if i1 is not None and i2 is not None:

                # If one of the objects is already matched in this image pair, discard.
                if i1 in matchesOf1 or i2 in matchesOf2:
                    logging.warning(
                        'One or two connected objects is already matched')
                    i1 = i2 = None

                else:
                    # Add the match to the list.
                    objectid1 = backendDb.objectField(objects1[i1], 'objectid')
                    objectid2 = backendDb.objectField(objects2[i2], 'objectid')
                    logging.debug('i1 = %d, i2 = %d' % (i1, i2))

                    # Check if this object already in the matches.
                    c.execute('SELECT match FROM matches WHERE objectid=?',
                              (objectid1, ))
                    matches1 = c.fetchall()
                    c.execute('SELECT match FROM matches WHERE objectid=?',
                              (objectid2, ))
                    matches2 = c.fetchall()

                    if len(matches1) > 1 or len(matches2) > 1:
                        logging.error(
                            'One of the objectids %d, %d is in several matches.'
                            % (objectid1, objectid2))
                        continue

                    elif len(matches1) == 1 and len(matches2) == 1:
                        logging.info(
                            'Will merge matches of objectids %d and %d' %
                            (objectid1, objectid2))
                        c.execute('UPDATE matches SET match=? WHERE match=?',
                                  (matches1[0][0], matches2[0][0]))

                    elif len(matches1) == 1 and len(matches2) == 0:
                        logging.info('Add objectid %d to match %d' %
                                     (objectid2, matches1[0][0]))
                        c.execute(
                            'INSERT INTO matches(match, objectid) VALUES (?,?)',
                            (matches1[0][0], objectid2))

                    elif len(matches1) == 0 and len(matches2) == 1:
                        logging.info('Add objectid %d to match %d' %
                                     (objectid1, matches2[0][0]))
                        c.execute(
                            'INSERT INTO matches(match, objectid) VALUES (?,?)',
                            (matches2[0][0], objectid1))

                    elif len(matches1) == 0 and len(matches2) == 0:
                        logging.info(
                            'Add a new match between objectids %d and %d.' %
                            (objectid1, objectid2))

                        # Find a free match index
                        c.execute('SELECT MAX(match) FROM matches')
                        maxmatch = c.fetchone()[0]
                        match = int(
                            maxmatch) + 1 if maxmatch is not None else 1

                        c.execute(
                            'INSERT INTO matches(match, objectid) VALUES (?,?)',
                            (match, objectid1))
                        c.execute(
                            'INSERT INTO matches(match, objectid) VALUES (?,?)',
                            (match, objectid2))

                    else:
                        assert False

                    # Reset when a new match is made.
                    needRedraw = True
                    i1 = i2 = None

            # Stay inside the loop inside one image pair until some button is pressed
            action = key_reader.parse(cv2.waitKey(50))

        # Process pressed key (all except exit)
        if action == 'previous':
            logging.info('Previous image pair')
            if index_image == 1:
                logging.warning('already the first image pair')
            else:
                index_image -= 1
        elif action == 'next':
            logging.info('Next image pair')
            index_image += 1  # exit at last image pair from outer loop
        elif action == 'delete_match':
            # if any car was selected, and it is matched
            if i1 is not None and i1 in matchesOf1:
                match = matchesOf1[i1]
                objectid1 = backendDb.objectField(objects1[i1], 'objectid')
                logging.info('deleting match %d' % match)
                c.execute(
                    'DELETE FROM matches WHERE match = ? AND objectid = ?',
                    (match, objectid1))
            else:
                logging.debug('delete is pressed, but no match is selected')

    cv2.destroyWindow("labelMatches")
