import os, os.path as op
import numpy as np
import regex as re  # Regex provides right-to-left parsing flag (?r)
import logging
import progressbar
import simplejson as json
import pprint
import tempfile
import shutil

from lib.backend import backendMedia
from lib.backend import backendDb
from lib.utils import util


def add_parsers(subparsers):
    upgradeStampImagepathsParser(subparsers)
    extractNumberIntoPropertyParser(subparsers)
    moveToRajaFolderStructureParser(subparsers)
    importNameFromCsvParser(subparsers)
    syncImagesWithDbParser(subparsers)
    validateImageNamesParser(subparsers)
    recordPositionOnPageParser(subparsers)
    getTop1NameParser(subparsers)
    setNumStampOccuranciesParser(subparsers)
    encodeNamesParser(subparsers)
    exportToJsonToPublishParser(subparsers)
    importJsonWithPredictionsParser(subparsers)
    classifyPagesParser(subparsers)


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
    for old_imagefile, in progressbar.progressbar(c.fetchall()):
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


def moveToRajaFolderStructureParser(subparsers):
    parser = subparsers.add_parser(
        'moveToRajaFolderStructure',
        description='Separate imagefiles by subfolders where they belong to '
        'according to Raja subfolder organization.')
    parser.set_defaults(func=moveToRajaFolderStructure)
    parser.add_argument('--subfolder_list_path',
                        default='testdata/stamps/subfolder_list.txt',
                        help="A path to the file with a list of subfolders.")
    target_dir_group = parser.add_mutually_exclusive_group()
    target_dir_group.add_argument(
        '--target_dir', help="Path to the folder with Raja's subfolders.")
    target_dir_group.add_argument(
        '--target_dims',
        choices=['1800x1200', '6Kx4K'],
        help='If specified, use default target directory for this dims.')
    parser.add_argument(
        '--rootdir_for_validation',
        help='If specified, will check that the image exists using the rootdir.'
    )


def moveToRajaFolderStructure(c, args):

    # "target_dir" may be set via "target_dims"
    if args.target_dir is not None:
        target_dir = args.target_dir
    elif args.target_dims is not None:
        target_dir = ('original_dataset/' if args.target_dims == '6Kx4K' else
                      '../etoropov/data/1800x1200/')
    else:
        assert False

    if target_dir[-1] != '/':
        logging.warning('Adding a "/" on the end of "%s".', target_dir)
        target_dir[-1] += '/'

    if not op.exists(args.subfolder_list_path):
        raise FileNotFoundError('File at subfolder_list_path not found: "%s"' %
                                args.subfolder_list_path)

    with open(args.subfolder_list_path) as f:
        subfolder_list = f.read().splitlines()

    c.execute('SELECT imagefile FROM images')
    for imagefile, in c.fetchall():
        imagename = op.basename(imagefile)
        # The extensions in Raja's folder are "JPG", not "jpg".
        imagename = op.splitext(imagename)[0] + '.JPG'
        # The number is the first two digits.
        subfolder_num = imagename[:2]
        try:
            subfolder = next(
                filter(lambda x: x[:2] == subfolder_num, subfolder_list))
        except StopIteration:
            raise ValueError('No subfolder starts with "%s"' %
                             subfolder_num) from None
        new_imagefile = op.join(target_dir, subfolder, imagename)
        logging.debug('Changing "%s" to "%s".', imagefile, new_imagefile)

        # Make sure that the file exists.
        if args.rootdir_for_validation:
            if not op.join(args.rootdir_for_validation, new_imagefile):
                raise FileNotFoundError(
                    'Transformed "%s" to "%s", but cant find this image '
                    'during validation with rootdir "%s".' %
                    (imagefile, new_imagefile, args.rootdir_for_validation))

        # Check if we used the database already with Raja's folder format.
        if imagefile == new_imagefile:
            raise ValueError('Old and new imagefiles are the same. '
                             'The database is already in Raja format.')

        c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, imagefile))
        c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, imagefile))


def importNameFromCsvParser(subparsers):
    parser = subparsers.add_parser(
        'importNameFromCsv',
        description='Imports name and score to objects. '
        'There can be several names with corresponding scores. In that case, '
        'both names and scores should be separated by comma: dog,cat 0.65,0.43'
    )
    parser.set_defaults(func=importNameFromCsv)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--delimiter', default='\t')
    parser.add_argument('--col_objectid', type=int, required=True)
    parser.add_argument('--col_name', type=int, required=True)
    parser.add_argument('--col_score', type=int, required=True)
    parser.add_argument(
        '--score_threshold',
        type=float,
        help='Only import rows with scores higher than threshold.')
    parser.add_argument('--bad_names',
                        nargs='*',
                        help='Do not import bad names if any.')


def importNameFromCsv(c, args):
    if not op.exists(args.csv_path):
        raise FileNotFoundError('File does not exist at: %s' % args.csv_path)
    rows = np.genfromtxt(args.csv_path,
                         delimiter=args.delimiter,
                         dtype=None,
                         encoding='UTF-8')

    num_names_per_object = {}
    logging.info('Found %d rows', len(rows))
    errors = 0
    for irow, row in progressbar.progressbar(enumerate(rows)):
        objectid = int(row[args.col_objectid])
        names = util.maybeDecode(row[args.col_name]).split(',')
        scores = [float(score) for score in row[args.col_score].split(',')]
        logging.debug('Names: %s, scores: %s', str(names), str(scores))
        if len(names) != len(scores):
            errors += 1
            raise ValueError(
                'Number of names "%s" and scores "%s" mismatches in row %d' %
                (names, scores, irow))
        names_and_scores = [(name, score)
                            for name, score in zip(names, scores)
                            if score >= args.score_threshold
                            and not (args.bad_names and name in args.bad_names)
                            ]
        names, scores = zip(
            *names_and_scores) if len(names_and_scores) > 0 else ([], [])
        logging.debug('Found %d names for object %d', len(names), objectid)

        c.execute('SELECT COUNT(1) FROM objects WHERE objectid=?',
                  (objectid, ))
        if c.fetchone()[0] == 0:
            raise ValueError("Didn't find objectid %d in the db." % objectid)

        name = ' / '.join(names)
        score = sum(scores)
        c.execute('UPDATE objects SET name=?,score=? WHERE objectid=?',
                  (name, score, objectid))

        if len(names) not in num_names_per_object:
            num_names_per_object[len(names)] = 0
        num_names_per_object[len(names)] += 1
    logging.info('Number of names per object:\n%s',
                 pprint.pformat(num_names_per_object))
    logging.info('Errors: %d', errors)


def syncImagesWithDbParser(subparsers):
    parser = subparsers.add_parser(
        'syncImagesWithDb',
        description='Populate "images" table with entries of ref_db. '
        'Also match objects by "objectid", and replace "imagefile" field '
        'for each object with its value from ref_db. '
        'Used to move to the original db after using tileObjects subcommand.')
    parser.set_defaults(func=syncImagesWithDb)
    parser.add_argument('--ref_db_file', required=True)


def syncImagesWithDb(c, args):
    # Check the ref_db.
    logging.info("ref_db_file: \n\t%s", args.ref_db_file)
    if not op.exists(args.ref_db_file):
        raise FileNotFoundError('Ref db does not exist: %s' % args.ref_db_file)

    # Work around attached database getting locked by making its temp copy,
    # and deleteing it instead of detaching it after the subcommand completes.
    ref_db_file = tempfile.NamedTemporaryFile().name
    shutil.copyfile(args.ref_db_file, ref_db_file)

    c.execute('ATTACH ? AS "ref"', (ref_db_file, ))

    # Check that all objectids have a correspondance in the ref_db_file.
    c.execute('SELECT COUNT(1) FROM objects')
    num_objects = c.fetchone()[0]
    c.execute('SELECT COUNT(1) FROM objects o '
              'INNER JOIN ref.objects ro ON o.objectid = ro.objectid')
    num_objects_with_match = c.fetchone()[0]
    if num_objects != num_objects_with_match:
        raise ValueError(
            'Out of %d objectids in the db, only %d have a match in ref_db_file'
            % (num_objects, num_objects_with_match))

    # Update "imagefile" field in the objects table.
    c.execute('UPDATE objects SET imagefile = ('
              'SELECT ro.imagefile FROM ref.objects ro '
              'WHERE objectid = ro.objectid)')

    # Replace "images" table.
    c.execute("DELETE FROM images")
    c.execute("INSERT INTO images SELECT * FROM ref.images")

    # c.execute('DETACH DATABASE "ref"')  Does not work, gets locked.
    os.remove(ref_db_file)


def validateImageNamesParser(subparsers):
    parser = subparsers.add_parser(
        'validateImageNames',
        description='Replace special characters in "imagefile" '
        'with the provided "replacement" cmd argument.')
    parser.set_defaults(func=validateImageNames)
    parser.add_argument('--replacement', default="_")


def validateImageNames(c, args):
    for original in ['+', '*', '?', '/']:
        c.execute('UPDATE images SET imagefile=REPLACE(imagefile, ?, ?)',
                  (original, args.replacement))
        c.execute('UPDATE objects SET imagefile=REPLACE(imagefile, ?, ?)',
                  (original, args.replacement))


def recordPositionOnPageParser(subparsers):
    parser = subparsers.add_parser(
        'recordPositionOnPage',
        description='Add information about the position of a stamp '
        'in its page to the properties table.')
    parser.set_defaults(func=recordPositionOnPage)
    parser.add_argument('--margin',
                        type=float,
                        default=20,
                        help='If no page is found, repeat with this margin.')


def recordPositionOnPage(c, args):
    # Clean old positions.
    c.execute('DELETE FROM properties WHERE key IN '
              '("x_on_page", "width_on_page", "y_on_page", "height_on_page")')
    # Get all stamp objects.
    c.execute('SELECT objectid,imagefile,x1+width/2,y1+height/2,width,height '
              'FROM objects WHERE name NOT LIKE "%page%"')
    for objectid, imagefile, x_stamp, y_stamp, width_stamp, height_stamp in (
            c.fetchall()):
        # Get pages in the same image with the stamp center inside their bbox.
        c.execute(
            'SELECT x1,y1,width,height FROM objects '
            'WHERE imagefile=? AND name LIKE "%page%" AND '
            'x1 < ? AND y1 < ? AND x1 + width > ? AND y1 + height > ?',
            (imagefile, x_stamp, y_stamp, x_stamp, y_stamp))
        pages = c.fetchall()

        if len(pages) == 0:
            # Try again with a margin.
            c.execute(
                'SELECT x1,y1,width,height FROM objects '
                'WHERE imagefile=? AND name LIKE "%page%" AND '
                'x1 - ? < ? AND x1 + width + ? > ? AND '
                'y1 - ? < ? AND y1 + height + ? > ?',
                (imagefile, args.margin, x_stamp, args.margin, x_stamp,
                 args.margin, y_stamp, args.margin, y_stamp))
            pages = c.fetchall()

            if len(pages) == 0:
                logging.error('Did not find a any page for stamp %d', objectid)
                x_perc = None
                y_perc = None
                width_perc = None
                height_perc = None

            else:
                logging.info('Found a page for stamp %d with margin', objectid)

        else:
            if len(pages) > 1:
                logging.warning('Found several pages with stamp %d', objectid)
            page = pages[0]

            # Get position and dimensions of a stamp relative to its page.
            x1_page, y1_page, width_page, height_page = page
            x_perc = (x_stamp - x1_page) / float(width_page)
            y_perc = (y_stamp - y1_page) / float(height_page)
            width_perc = width_stamp / float(width_page)
            height_perc = height_stamp / float(width_page)

            assert 0 <= x_perc <= 1, x_perc
            assert 0 <= y_perc <= 1, y_perc

        if len(pages) > 0:
            c.execute(
                'INSERT INTO properties(objectid, key, value) VALUES '
                '(?, "x_on_page", ?), (?, "width_on_page", ?), '
                '(?, "y_on_page", ?), (?, "height_on_page", ?)',
                (objectid, str(x_perc), objectid, str(width_perc), objectid,
                 str(y_perc), objectid, str(height_perc)))


def getTop1NameParser(subparsers):
    parser = subparsers.add_parser(
        'getTop1Name',
        description='Replace all names like "cat / dog / sheep" with "cat". '
        'The first guess is assumed to be first. Guesses are assumed to be '
        'separated by " / ".')
    parser.set_defaults(func=getTop1Name)


def getTop1Name(c, args):
    c.execute("UPDATE objects SET name = substr(name, 0, instr(name, ' / ')) "
              "WHERE name LIKE '% / %'")


def setNumStampOccuranciesParser(subparsers):
    parser = subparsers.add_parser(
        'setNumStampOccurancies',
        description='Adds or updates property "num_instances" which tells how '
        'many stamps there are of this name. This value is obviously the same '
        'for stamps with the same name.')
    parser.set_defaults(func=setNumStampOccurancies)


def setNumStampOccurancies(c, args):
    c.execute('DELETE FROM properties WHERE key = "num_instances"')
    c.execute(
        'INSERT INTO properties(objectid,key,value) SELECT o1.objectid, "num_instances", '
        '(SELECT CAST(COUNT(1) AS TEXT) FROM objects WHERE o1.name = name) FROM objects o1'
    )


def encodeNamesParser(subparsers):
    parser = subparsers.add_parser(
        'encodeNames',
        description='Encode names to integers. Write encoding to json.')
    parser.set_defaults(func=encodeNames)
    parser.add_argument('--encoding_json_file',
                        required=True,
                        help='Mapping from name to label.')


def encodeNames(c, args):
    c.execute('SELECT name FROM objects WHERE name NOT LIKE "%??%" AND '
              'name NOT LIKE "page%" GROUP BY name ORDER BY COUNT(1) DESC')
    name_list = [x for x, in c.fetchall()]
    name_to_id = {name: name_id for name_id, name in enumerate(name_list)}

    # First, remove whatever is in there.
    c.execute('DELETE FROM properties WHERE key="name_id"')

    # Add the special case to the encoding.
    c.execute('SELECT name FROM objects WHERE name LIKE "%??%"')
    name_list = [x for x, in c.fetchall()]
    for name in name_list:
        name_to_id[name] = -1

    with open(args.encoding_json_file, 'w') as f:
        json.dump(name_to_id, f, indent=4)

    c.execute('SELECT objectid,name FROM objects')
    for objectid, name in c.fetchall():
        if name in name_to_id:
            name_id = name_to_id[name]
        else:
            logging.debug('Skip objectid %d with name "%s"', objectid, name)
            continue
        logging.debug('Writing name "%s" under id %d for objectid %d.', name,
                      name_id, objectid)
        c.execute(
            'INSERT INTO properties(objectid,key,value) VALUES (?,"name_id",?)',
            (objectid, str(name_id)))


def exportToJsonToPublishParser(subparsers):
    parser = subparsers.add_parser(
        'exportToJsonToPublish',
        description='Export fields and properties to json.')
    parser.set_defaults(func=exportToJsonToPublish)
    parser.add_argument('--add_name', action='store_true')
    parser.add_argument('--add_score', action='store_true')
    parser.add_argument('--add_imagefile', action='store_true')
    parser.add_argument('--add_objectid', action='store_true')
    parser.add_argument('--keys',
                        nargs='+',
                        required=True,
                        help='The keys of properties to export.')
    parser.add_argument(
        '--value_types',
        nargs='+',
        choices=['str', 'int', 'float'],
        help='Optional types of properties values. '
        'The list of "value_types" must match the list of "keys". '
        'If provided, values will be parsed as this type.')
    parser.add_argument(
        '--json_path',
        help='If specified, write to json. Otherwise, to stdout.')


def exportToJsonToPublish(c, args):
    results = {'objects': []}

    if args.value_types is not None and len(args.keys) != len(
            args.value_types):
        raise ValueError('"value_types" specified, but its length %d '
                         'does not match the length of "keys" %d' %
                         (len(args.value_types), len(args.keys)))
    elif args.value_types is None:
        args.value_types = ['str'] * len(args.keys)
    transforms = {}
    for key, type_str in zip(args.keys, args.value_types):
        if type_str == 'str':
            transforms[key] = lambda x: x
        elif type_str == 'int':
            transforms[key] = lambda x: int(x)
        elif type_str == 'float':
            transforms[key] = lambda x: float(x)
        else:
            assert 0, 'We should not be here with type %s' % type_str

    keys_list = ','.join(['"%s"' % key for key in args.keys])
    properties_query = 'SELECT key,value FROM properties WHERE objectid=? AND key IN (%s)' % keys_list
    logging.debug(properties_query)

    c.execute(
        'SELECT objectid,imagefile,name,score FROM objects ORDER BY imagefile')
    for objectid, imagefile, name, score in c.fetchall():
        c.execute(properties_query, (objectid, ))
        entries = dict(c.fetchall())
        logging.debug(entries)
        # Apply transform to the value.
        result = {k: transforms[k](v) for k, v in entries.items()}
        logging.debug(result)

        if args.add_objectid:
            if 'objectid' in result:
                raise ValueError('Cant add object name to results, because '
                                 '"objectid" is already a key of a property.')
            result['objectid'] = objectid

        if args.add_name:
            if 'name' in result:
                raise ValueError('Cant add object name to results, because '
                                 '"name" is already a key of a property.')
            result['name'] = name

        if args.add_score:
            if 'score' in result:
                raise ValueError('Cant add object name to results, because '
                                 '"score" is already a key of a property.')
            result['score'] = score

        if args.add_imagefile:
            if 'imagefile' in result:
                raise ValueError('Cant add object name to results, because '
                                 '"imagefile" is already a key of a property.')
            result['imagefile'] = imagefile

        results['objects'].append(result)

    if args.json_path:
        with open(args.json_path, 'w') as f:
            f.write(json.dumps(results, sort_keys=True, indent=2))
    else:
        print(json.dumps(results, sort_keys=True, indent=2))


def importJsonWithPredictionsParser(subparsers):
    parser = subparsers.add_parser(
        'importJsonWithPredictions',
        description='Export fields and properties to json.')
    parser.set_defaults(func=importJsonWithPredictions)
    parser.add_argument(
        '--json_file',
        required=True,
        help='If specified, write to json. Otherwise, to stdout.')
    parser.add_argument('--encoding_json_file',
                        required=True,
                        help='Mapping from name to label.')


def importJsonWithPredictions(c, args):
    with open(args.encoding_json_file) as f:
        encoding = json.load(f)
    decoding = {}
    for name, name_id in encoding.items():
        if name_id == -1:
            decoding[name_id] = None
        elif name_id in decoding:
            raise ValueError('Not expecting multiple back mapping.')
        else:
            decoding[name_id] = name
    logging.info('Have %d entries in decoding.', len(decoding))

    with open(args.json_file) as f:
        data = json.load(f)
    c.execute('DELETE FROM properties WHERE key="classification_score"')
    c.execute('DELETE FROM properties WHERE key="classification_name_id"')
    for objectid in progressbar.progressbar(data['objects']):
        object_ = data['objects'][objectid][0]
        objectid = int(objectid)
        name_id = object_['classification_name_ids'][0]
        score = object_['classification_scores'][0]
        c.execute(
            'INSERT INTO properties(objectid,key,value) '
            'VALUES(?,"classification_score",?)', (objectid, str(score)))
        c.execute(
            'INSERT INTO properties(objectid,key,value) '
            'VALUES(?,"classification_name_id",?)', (objectid, str(name_id)))
        if name_id not in decoding:
            raise ValueError('name_id %d not in decoding.')
        c.execute('UPDATE objects SET name=? WHERE objectid=?',
                  (decoding[name_id], objectid))


def classifyPagesParser(subparsers):
    parser = subparsers.add_parser(
        'classifyPages',
        description=
        'Classify pages into right and left. Only cares about name "page".')
    parser.set_defaults(func=classifyPages)
    parser.add_argument(
        '--IoU_threshold',
        type=int,
        default=0.2,
        help=
        'Pages are classified if the x-axis IoU is less than this threshold.')


def classifyPages(c, args):
    count = 0

    c.execute('SELECT imagefile FROM images')
    for imagefile, in c.fetchall():
        c.execute(
            'SELECT objectid,x1,x1+width FROM objects '
            'WHERE imagefile=? AND name="page"', (imagefile, ))

        page_entries = c.fetchall()
        logging.debug('Found %d pages for %s', len(page_entries), imagefile)

        if len(page_entries) != 2:
            continue

        # Calculate IoU.
        objectid_a, x1_a, x2_a = page_entries[0]
        objectid_b, x1_b, x2_b = page_entries[1]
        intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
        union = (x2_a - x1_a) + (x2_b - x1_b) - intersection
        IoU = float(intersection) / union
        assert IoU >= 0, (intersection, union)
        logging.debug('x1_a: %d, x2_a: %d, x1_b: %d, x2_b: %d, IoU: %0.3f',
                      x1_a, x2_a, x1_b, x2_b, IoU)

        if IoU < args.IoU_threshold:
            count += 1
            if x1_a + x2_a < x1_b + x2_b:
                objectid_left = objectid_a
                objectid_right = objectid_b
            else:
                objectid_left = objectid_b
                objectid_right = objectid_a
            c.execute('UPDATE objects SET name="page_l" WHERE objectid=?',
                      (objectid_left, ))
            c.execute('UPDATE objects SET name="page_r" WHERE objectid=?',
                      (objectid_right, ))

    logging.info('Have updated %d images.', count)
