import os, os.path as op
import numpy as np
import regex as re  # Regex provides right-to-left parsing flag (?r)
import logging
from progressbar import progressbar
import simplejson as json

from lib.backend import backendMedia
from lib.backend import backendDb
from lib.utils import util


def add_parsers(subparsers):
    upgradeStampImagepathsParser(subparsers)
    extractNumberIntoPropertyParser(subparsers)


def upgradeStampImagepathsParser(subparsers):
    parser = subparsers.add_parser(
        'upgradeStampImagepaths',
        description=
        'Change imagefile from an old to a new location for every image. '
        'All new imagefiles must be in the same directory.'
        'Paths are changed for each image based on a provided correspondence map. '
        'If the image at an old path is not almost-the-same as the image at an new path '
        'an exception is thrown. '
        'Imagefile without a correspondence are changed to NULL.')
    parser.add_argument('--new_image_dir',
                        required=True,
                        help='A directory where all new images are.')
    parser.add_argument(
        '--name_correspondences_json_file',
        required=True,
        help='A json file with a map <old_image_name>: <new_image_name>.')
    parser.add_argument(
        '--check_if_images_are_close',
        action='store_true',
        help='If specified, the new imagefile is verified to be pixelwise '
        'similar to old imagefile.')
    parser.set_defaults(func=upgradeStampImagepaths)


def upgradeStampImagepaths(c, args):
    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    with open(args.name_correspondences_json_file) as f:
        old_to_new_path_dict = json.load(f)
    logging.info('Found %d pairs in the json file.', len(old_to_new_path_dict))

    c.execute('SELECT imagefile FROM images')
    for old_imagefile, in progressbar(c.fetchall()):
        old_name = op.basename(old_imagefile)
        if old_name not in old_to_new_path_dict:
            logging.debug('Old name %s does NOT have a match. Deleting.',
                          old_name)
            backendDb.deleteImage(c, old_imagefile)
            continue

        new_name = old_to_new_path_dict[old_name]
        logging.debug('Old name %s is matched to new name %s', old_name,
                      new_name)

        new_imagefile = op.join(args.new_image_dir, new_name)

        if args.check_if_images_are_close:
            old_image = imreader.imread(old_imagefile)
            new_image = imreader.imread(new_imagefile)

            diff = np.abs(old_image.astype(float) -
                          new_image.astype(float)).sum()
            avg = (old_image.astype(float) +
                   new_image.astype(float)).sum() / 2.
            perc_diff = diff / avg

            THRESH = 0.05  # 5 percent.
            if perc_diff > THRESH:
                raise ValueError(
                    'Image at old_imagefile "%s" is different from image '
                    'at new_imagefile "%s" by %.3f %%.' %
                    (old_imagefile, new_imagefile, perc_diff))

        c.execute('SELECT COUNT(1) FROM images WHERE imagefile=?',
                  (new_imagefile, ))
        if c.fetchone()[0] > 0:
            # If a duplicate new imagefile, merge objects, but delete the image.
            logging.warning(
                'Removing a duplicate new image "%s". '
                'Objects will be merged', new_imagefile)
            c.execute('DELETE FROM images WHERE imagefile=?',
                      (old_imagefile, ))
        else:
            # If not a duplicate, rename imagefile.
            c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                      (new_imagefile, old_imagefile))

        c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, old_imagefile))


def extractNumberIntoPropertyParser(subparsers):
    parser = subparsers.add_parser(
        'extractNumberIntoProperty',
        description='Cuts out number from object name field, '
        'and assigns it to a new property with user-specified name. '
        'Currently ignores all but the first number in the name, if any.')
    parser.set_defaults(func=extractNumberIntoProperty)
    parser.add_argument('--property',
                        required=True,
                        help='Property name to assign number to.')


def extractNumberIntoProperty(c, args):
    c.execute('SELECT objectid,name FROM objects')
    for objectid, name in c.fetchall():
        name = util.maybeDecode(name)
        logging.debug('name: %s', name)
        if name is None:
            continue
        # Flag (?r) for parsing right-to-left.
        for match in re.finditer(r'(?r)(\d+)', name):
            number = match.group()
            span = match.span()
            logging.debug('From name "%s" got match "%s" spanning (%d,%d)',
                          name, number, span[0], span[1])
            c.execute(
                'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                (objectid, args.property, number))
            c.execute(
                'UPDATE objects SET name = SUBSTR(name,1,?) || SUBSTR(name,?+1) '
                'WHERE objectid = ?', (span[0], span[1], objectid))
