import os.path as op
import numpy as np
import cv2
import logging
import progressbar
import pprint
import ast
import datetime
import math

from lib.backend import backendMedia
from lib.backend import backendDb
from lib.utils import utilBoxes
from lib.utils import util


def add_parsers(subparsers):
    writeMediaParser(subparsers)
    cropMediaParser(subparsers)
    cropObjectsParser(subparsers)
    polygonsToMaskParser(subparsers)
    writeMediaGridByTimeParser(subparsers)
    repaintMaskParser(subparsers)
    tileObjectsParser(subparsers)


def cropMediaParser(subparsers):
    parser = subparsers.add_parser('cropMedia',
                                   description='Crops images to a single ROI.')
    parser.set_defaults(func=cropMedia)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='Output either a directory with pictures or a video file.')
    parser.add_argument(
        '--out_rootdir',
        help='Specify, if rootdir changed for the output imagery.')
    parser.add_argument('--image_path',
                        required=True,
                        help='The directory for pictures OR video file, '
                        'where crops from images are written to.')
    parser.add_argument('--mask_path',
                        help='the directory for pictures OR video file, '
                        'where crops from masks are written to.')
    parser.add_argument('--x1', type=int, required=True)
    parser.add_argument('--y1', type=int, required=True)
    parser.add_argument('--x2', type=int, required=True)
    parser.add_argument('--y2', type=int, required=True)
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite video if it exists.')


def cropMedia(c, args):
    imreader = backendMedia.MediaReader(rootdir=args.rootdir)
    # Create a writer. Rootdir may be changed.
    out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
    imwriter = backendMedia.MediaWriter(rootdir=out_rootdir,
                                        media_type=args.media,
                                        image_media=args.image_path,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    target_width = args.x2 - args.x1
    target_height = args.y2 - args.y1

    c.execute('SELECT * FROM images')
    for image_entry in progressbar.progressbar(c.fetchall()):
        imagefile = backendDb.imageField(image_entry, 'imagefile')
        maskfile = backendDb.imageField(image_entry, 'maskfile')

        image = imreader.imread(imagefile)
        image = image[args.y1:args.y2, args.x1:args.x2, :]
        if args.mask_path is not None and maskfile is not None:
            mask = imreader.maskread(maskfile)
            mask = mask[args.y1:args.y2, args.x1:args.x2]

        # Write an image.
        if args.image_path is not None:
            imagefile_new = imwriter.imwrite(image)
        else:
            imagefile_new = None

        # Write mask.
        if args.mask_path is not None and maskfile is not None:
            maskfile_new = imwriter.maskwrite(mask)
        else:
            maskfile_new = None

        # Update the database entry.
        if maskfile_new is not None:
            c.execute(
                'UPDATE images SET maskfile=?,width=?,height=? WHERE imagefile=?',
                (maskfile_new, target_width, target_height, imagefile))
        if imagefile_new is not None:
            c.execute(
                'UPDATE images SET imagefile=?,width=?,height=? WHERE imagefile=?',
                (imagefile_new, target_width, target_height, imagefile))
            c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                      (imagefile_new, imagefile))

    c.execute(
        'UPDATE objects SET x1=(SELECT x1 FROM objects)-?, y1=(SELECT y1 FROM objects)-?',
        (args.x1, args.y1))
    # TODO: update polygons.
    imwriter.close()


def cropObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'cropObjects',
        description=
        'Crops object patches to pictures or video and saves their info as a db. '
        'All imagefiles and maskfiles in the db will be replaced with crops, '
        'one frame/image per object.')
    parser.set_defaults(func=cropObjects)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='output either a directory with pictures or a video file.')
    parser.add_argument('--image_path',
                        required=True,
                        help='The directory for pictures OR video file, '
                        'where cropped images are written to.')
    parser.add_argument('--mask_path',
                        help='the directory for pictures OR video file, '
                        'where cropped masks are written to.')
    parser.add_argument(
        '--where_object',
        default='TRUE',
        help='SQL "where" clause that specifies which objects to crop. '
        'Queries table "objects". Example: \'objects.name == "car"\'')
    parser.add_argument(
        '--where_other_objects',
        default='FALSE',
        help='SQL "where" clause that specifies which other objects from '
        'a frame to write for each crop. '
        'By default, only the cropped object is recorded.'
        'Queries table "objects". Example: \'objects.name == "bus"\'')
    parser.add_argument('--target_width', type=int)
    parser.add_argument('--target_height', type=int)
    parser.add_argument(
        '--edges',
        default='distort',
        choices={'distort', 'constant', 'background', 'original'},
        help='"distort" distorts the patch to get to the desired ratio, '
        '"constant" keeps the ratio but pads the patch with zeros, '
        '"background" keeps the ratio but includes image background. '
        '"original" does not change the crop dimensions. Target width and height are ignored.'
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite video if it exists.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--split_into_folders_by_object_name',
        action='store_true',
        help=
        'Images are split into folders by object name, if media=""pictures".')
    group.add_argument(
        '--add_object_name_to_filename',
        action='store_true',
        help='Object name is added to image name, if media=""pictures".')


def cropObjects(c, args):
    imreader = backendMedia.MediaReader(rootdir=args.rootdir)
    imwriter = backendMedia.MediaWriter(media_type=args.media,
                                        rootdir=args.rootdir,
                                        image_media=args.image_path,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    backendDb.retireTables(c)

    c.execute(
        'SELECT '
        'objects.objectid,'
        'objects.imagefile,'
        'objects.x1,'
        'objects.y1,'
        'objects.width,'
        'objects.height,'
        'objects.name,'
        'objects.score,'
        'images.maskfile,'
        'images.timestamp '
        'FROM objects_old AS objects '
        'INNER JOIN images_old AS images ON images.imagefile = objects.imagefile '
        'WHERE (%s) ORDER BY images.imagefile' % args.where_object)
    old_entries = c.fetchall()
    logging.debug(pprint.pformat(old_entries))

    # Avoid unnecessary reloading the same image if imagefile = prev_imagefile.
    prev_old_imagefile = None

    for old_objectid, old_imagefile, old_x1, old_y1, old_width, old_height, \
        name, score, old_maskfile, timestamp in progressbar.progressbar(old_entries):
        logging.debug('Processing object %d from imagefile %s.', old_objectid,
                      old_imagefile)
        old_roi = utilBoxes.bbox2roi((old_x1, old_y1, old_width, old_height))

        # Write image.
        if prev_old_imagefile != old_imagefile:
            old_image = imreader.imread(old_imagefile)
        logging.debug('Cropping roi=%s from image of shape %s', old_roi,
                      old_image.shape)
        new_image, transform = utilBoxes.cropPatch(old_image, old_roi,
                                                   args.edges,
                                                   args.target_height,
                                                   args.target_width)
        if args.split_into_folders_by_object_name:
            # TODO: Do something about names with spacial characters.
            #       Cant use validateFileName because two names may get mapped
            #       into one filename.
            namehint = '%s/%09d' % (util.maybeDecode(name), old_objectid)
        elif args.add_object_name_to_filename:
            # TODO: Ditto.
            namehint = '%s %09d' % (util.maybeDecode(name), old_objectid)
        else:
            namehint = '%09d' % old_objectid
        new_imagefile = imwriter.imwrite(new_image, namehint=namehint)
        new_width, new_height = new_image.shape[1], new_image.shape[0]

        # Write transform as x_new = x_old * kx + bx, y_new = y_old * ky + by.
        ky = transform[0, 0]
        kx = transform[1, 1]
        by = transform[0, 2]
        bx = transform[1, 2]

        # Write mask.
        if args.mask_path is not None and old_maskfile is not None:
            if prev_old_imagefile != old_imagefile:
                old_mask = imreader.maskread(old_maskfile)
            new_mask, _ = utilBoxes.cropPatch(old_mask, old_roi, args.edges,
                                              args.target_height,
                                              args.target_width)
            new_maskfile = imwriter.maskwrite(new_mask, namehint=namehint)
        else:
            new_maskfile = None

        # Insert image values.
        logging.debug('Recording imagefile %s and maskfile %s.', new_imagefile,
                      new_maskfile)
        c.execute(
            'INSERT INTO images(imagefile, width, height, maskfile, timestamp, name, score) '
            'VALUES (?,?,?,?,?,?,?)', (new_imagefile, new_width, new_height,
                                       new_maskfile, timestamp, name, score))

        # Select the cropped object and "where_other_objects" objects in this image.
        c.execute(
            'SELECT objectid FROM objects_old WHERE objectid=? OR (imagefile=? AND (%s))'
            % args.where_other_objects, (old_objectid, old_imagefile))
        for old_im_objectid, in c.fetchall():
            # Insert to objects.
            c.execute(
                'INSERT INTO objects(imagefile,x1,y1,width,height,name,score) '
                'SELECT ?, x1 * ? + ?, y1 * ? + ?, width * ?, height * ?, name, score '
                'FROM objects_old WHERE objectid=?',
                (new_imagefile, kx, bx, ky, by, kx, ky, old_im_objectid))
            new_im_objectid = c.lastrowid
            # Insert to properties.
            c.execute(
                'INSERT INTO properties(objectid,key,value) '
                'SELECT ?,key,value FROM properties_old WHERE objectid=?',
                (new_im_objectid, old_im_objectid))
            backendDb.updateObjectTransform(c, new_im_objectid, transform)
            # Insert to polygons.
            c.execute(
                'INSERT INTO polygons(objectid,x,y) '
                'SELECT ?, x * ? + ?, y * ? + ? FROM polygons_old WHERE objectid=?',
                (new_im_objectid, kx, bx, ky, by, old_im_objectid))
            # Insert to matches.
            c.execute(
                'INSERT INTO matches(match,objectid) '
                'SELECT match,? FROM matches_old WHERE objectid=?',
                (new_im_objectid, old_im_objectid))
            # Add a property to the actual cropped object that says it is cropped.
            if old_im_objectid == old_objectid:
                c.execute(
                    'INSERT INTO properties(objectid,key,value) VALUES (?,?,?)',
                    (old_objectid, 'cropped', 'true'))

    backendDb.dropRetiredTables(c)
    imwriter.close()


def tileObjectsParser(subparsers):
    parser = subparsers.add_parser(
        'tileObjects',
        description=
        'Tile objects into collage images, each collage having YxX objects. '
        'Any objects in the original images are preserved, '
        'their bounding boxes and polygons adjusted accordingly. '
        'The purpose of the function is to create collages for the convenient '
        'inspection of objects. Masks are not supported at this moment.')
    parser.set_defaults(func=tileObjects)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='Output either a directory with pictures or a video file.')
    parser.add_argument('--image_path',
                        required=True,
                        help='The directory for pictures OR video file, '
                        'where cropped images are written to.')
    parser.add_argument(
        '--where_object',
        default='TRUE',
        help='SQL "where" clause that specifies which objects to crop. '
        'Queries table "objects". Example: \'objects.name == "car"\'')
    parser.add_argument('--num_cells_Y',
                        type=int,
                        default=5,
                        help='Number of objects in Y dimension.')
    parser.add_argument('--num_cells_X',
                        type=int,
                        default=5,
                        help='Number of objects in X dimension.')
    parser.add_argument('--inter_cell_gap',
                        type=int,
                        default=10,
                        help='Gap in between objects in pixels.')
    parser.add_argument('--cell_width', type=int, required=True)
    parser.add_argument('--cell_height', type=int, required=True)
    parser.add_argument(
        '--edges',
        default='constant',
        choices={'constant', 'background'},
        help='"constant" keeps the ratio but pads the patch with zeros, '
        '"background" keeps the ratio but includes image background.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite video if it exists.')
    parser.add_argument('--split_by_name',
                        action='store_true',
                        help='Start a new collage with every new name.')
    parser.add_argument(
        '--image_icon',
        action='store_true',
        help='If specified, adds an icon of the whole image with object white.'
    )


def tileObjects(c, args):
    # Temporary imagefile. See the use below.
    TEMP_IMAGEFILE = 'temp_imagefile'

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)
    imwriter = backendMedia.MediaWriter(media_type=args.media,
                                        rootdir=args.rootdir,
                                        image_media=args.image_path,
                                        overwrite=args.overwrite)

    backendDb.retireTables(c)

    c.execute(
        'SELECT '
        'objects.objectid,'
        'objects.imagefile,'
        'objects.x1,'
        'objects.y1,'
        'objects.width,'
        'objects.height,'
        'objects.name,'
        'objects.score,'
        'images.maskfile,'
        'images.timestamp '
        'FROM objects_old AS objects '
        'INNER JOIN images_old AS images ON images.imagefile = objects.imagefile '
        'WHERE (%s) ORDER BY objects.name' % args.where_object)
    old_entries = c.fetchall()
    logging.debug(pprint.pformat(old_entries))

    # Create collages and info about them.
    cell_width = args.cell_width * 2 if args.image_icon else args.cell_width
    num_cells_per_collage = args.num_cells_Y * args.num_cells_X
    if num_cells_per_collage == 0:
        raise ValueError('Need num_cells_Y > 0 and num_cells_X > 0.')
    gap = args.inter_cell_gap  # Convenience alias.
    collage_X = args.num_cells_X * (cell_width + gap) - gap
    collage_Y = args.num_cells_Y * (args.cell_height + gap) - gap

    def _recordCollage(c, collage, namehint):
        new_imagefile = imwriter.imwrite(
            collage, namehint=util.validateFileName(namehint))
        # Insert image values.
        logging.debug('Recording imagefile with namehint: %s', namehint)
        logging.info('Recording imagefile %s', new_imagefile)
        c.execute(
            'INSERT INTO images(imagefile, width, height, timestamp, name, score) '
            'VALUES (?,?,?,?,?,?)',
            (new_imagefile, collage_X, collage_Y, timestamp, name, score))
        # Before now objects were written to TEMP_IMAGEFILE
        c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                  (new_imagefile, TEMP_IMAGEFILE))

    # Bookkeeping.
    previous_name = None
    i_cell = 0
    i_collage = 0
    collage = np.zeros((collage_Y, collage_X, 3), dtype=np.uint8)

    for old_entry in progressbar.progressbar(old_entries):
        # Extract all the info from 'old_entry'.
        (objectid, old_imagefile, old_x1, old_y1, old_width, old_height, name,
         score, _, timestamp) = old_entry
        logging.debug('Processing object %d from imagefile %s.', objectid,
                      old_imagefile)
        if old_width * old_height == 0:
            raise ValueError('objectid %d is degenerate with size: %dx%d' %
                             (objectid, old_width, old_height))
        old_roi = utilBoxes.bbox2roi((old_x1, old_y1, old_width, old_height))

        # Record the collage when the previous collage is full or name changed.
        if i_cell == num_cells_per_collage - 1 or (args.split_by_name
                                                   and previous_name != name):
            namehint = '%09d' % i_collage
            if args.split_by_name:
                namehint += '-%s' % previous_name
            _recordCollage(c, collage, namehint)
            collage = np.zeros((collage_Y, collage_X, 3), dtype=np.uint8)
            i_cell = -1
            i_collage += 1
        i_cell += 1
        previous_name = name

        # Crop object.
        old_image = imreader.imread(old_imagefile)
        logging.debug('Cropping roi=%s from image of shape %s', old_roi,
                      old_image.shape)
        crop, transform = utilBoxes.cropPatch(old_image, old_roi, args.edges,
                                              args.cell_height,
                                              args.cell_width)

        # A small copy of the image.
        if args.image_icon:
            old_image[
                max(old_roi[0], 0):min(old_roi[2], old_image.shape[0]),
                max(old_roi[1], 0):min(old_roi[3], old_image.shape[1])] = 255
            image_icon, _ = utilBoxes.cropPatch(
                old_image, [0, 0, old_image.shape[0], old_image.shape[1]],
                'constant', args.cell_height, args.cell_width)
            crop = np.hstack([crop, image_icon])

        # Get the cell coordinates. Cells are populated row by row.
        cell_x = (i_cell % args.num_cells_X) * (cell_width + gap)
        cell_y = (i_cell % num_cells_per_collage //
                  args.num_cells_X) * (args.cell_height + gap)
        logging.debug('i_cell: %d, cell_x: %d, cell_y: %d', i_cell, cell_x,
                      cell_y)

        assert cell_y + args.cell_height <= collage_Y, i_cell
        assert cell_x + cell_width <= collage_X, i_cell
        collage[cell_y:cell_y + args.cell_height,
                cell_x:cell_x + cell_width] = crop

        transform[0, 2] += cell_y
        transform[1, 2] += cell_x

        # Write transform as x_new = x_old * kx + bx, y_new = y_old * ky + by.
        ky = transform[0, 0]
        kx = transform[1, 1]
        by = transform[0, 2]
        bx = transform[1, 2]

        # Insert to objects.
        c.execute(
            'INSERT INTO objects(objectid,imagefile,x1,y1,width,height,name,score) '
            'SELECT ?, ?, x1 * ? + ?, y1 * ? + ?, width * ?, height * ?, name, score '
            'FROM objects_old WHERE objectid=?',
            (objectid, TEMP_IMAGEFILE, kx, bx, ky, by, kx, ky, objectid))
        # Insert to properties.
        c.execute(
            'INSERT INTO properties(objectid,key,value) SELECT ?,"old_imagefile",?',
            (objectid, old_imagefile))
        c.execute(
            'INSERT INTO properties(objectid,key,value) '
            'SELECT ?,key,value FROM properties_old WHERE objectid=?',
            (objectid, objectid))
        backendDb.updateObjectTransform(c, objectid, transform)
        # Insert to polygons.
        c.execute(
            'INSERT INTO polygons(objectid,x,y) '
            'SELECT ?, x * ? + ?, y * ? + ? FROM polygons_old WHERE objectid=?',
            (objectid, kx, bx, ky, by, objectid))
        # Insert to matches.
        c.execute(
            'INSERT INTO matches(match,objectid) '
            'SELECT match,? FROM matches_old WHERE objectid=?',
            (objectid, objectid))

    # Record the last (partially filled) collage.
    namehint = '%09d' % i_collage
    if args.split_by_name:
        namehint += '-%s' % previous_name
    _recordCollage(c, collage, namehint)

    backendDb.dropRetiredTables(c)
    imwriter.close()


def writeMediaParser(subparsers):
    parser = subparsers.add_parser(
        'writeMedia',
        description='Export images as a directory with pictures or as a video, '
        'and change the database imagefiles and maskfiles to match the recordings.'
    )
    parser.set_defaults(func=writeMedia)
    parser.add_argument(
        '--out_rootdir',
        help='Specify, if rootdir changed for the output imagery.')
    parser.add_argument('--where_image',
                        default='TRUE',
                        help='SQL where clause for the "images" table.')
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='output either a directory with pictures or a video file.')
    parser.add_argument(
        '--image_path',
        help=
        'The directory for pictures OR video file, where images are written to.'
    )
    parser.add_argument(
        '--mask_path',
        help=
        'The directory for pictures OR video file, where masks are written to.'
    )
    parser.add_argument('--mask_mapping_dict',
                        help='How values in maskfile are drawn. '
                        'E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mask_aside',
                       action='store_true',
                       help='Draw image and mask side by side.')
    group.add_argument('--mask_alpha',
                       type=float,
                       help='Transparency to overlay the label mask with, '
                       '1 means cant see the image behind the mask.')
    parser.add_argument('--with_imageid',
                        action='store_true',
                        help='print frame number.')
    parser.add_argument('--with_objects',
                        action='store_true',
                        help='draw objects on top.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite video if it exists.')


def writeMedia(c, args):
    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    # For overlaying masks.
    labelmap = ast.literal_eval(
        args.mask_mapping_dict) if args.mask_mapping_dict else None
    logging.info('Parsed mask_mapping_dict to %s', pprint.pformat(labelmap))

    # Create a writer. Rootdir may be changed.
    out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
    imwriter = backendMedia.MediaWriter(rootdir=out_rootdir,
                                        media_type=args.media,
                                        image_media=args.image_path,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    logging.info('Writing imagery and updating the database.')
    c.execute(
        'SELECT imagefile,maskfile FROM images WHERE %s ORDER BY imagefile' %
        args.where_image)
    for imagefile, maskfile in progressbar.progressbar(c.fetchall()):

        logging.debug('Imagefile "%s"', imagefile)
        if args.image_path is not None:
            image = imreader.imread(imagefile)

        # Overlay the mask.
        if maskfile is not None:
            mask = imreader.maskread(maskfile)
            if args.image_path is not None:
                if args.mask_aside:
                    image = util.drawMaskAside(image, mask, labelmap=labelmap)
                elif args.mask_alpha is not None:
                    image = util.drawMaskOnImage(image,
                                                 mask,
                                                 alpha=args.mask_alpha,
                                                 labelmap=labelmap)
        else:
            mask = None
            logging.debug('No mask for this image.')

        # Overlay imagefile.
        if args.with_imageid:
            util.drawTextOnImage(image, op.basename(imagefile))

        # Draw objects as polygons (preferred) or ROI.
        if args.with_objects:
            c.execute('SELECT * FROM objects WHERE imagefile=?', (imagefile, ))
            object_entries = c.fetchall()
            logging.debug('Found %d objects', len(object_entries))
            for object_entry in object_entries:
                objectid = backendDb.objectField(object_entry, 'objectid')
                roi = backendDb.objectField(object_entry, 'roi')
                score = backendDb.objectField(object_entry, 'score')
                name = backendDb.objectField(object_entry, 'name')
                c.execute('SELECT * FROM polygons WHERE objectid=?',
                          (objectid, ))
                polygon_entries = c.fetchall()
                if len(polygon_entries) > 0:
                    logging.debug('showing object with a polygon.')
                    polygon = [(backendDb.polygonField(p, 'x'),
                                backendDb.polygonField(p, 'y'))
                               for p in polygon_entries]
                    util.drawScoredPolygon(image,
                                           polygon,
                                           label=name,
                                           score=score)
                elif roi is not None:
                    logging.debug('showing object with a bounding box.')
                    util.drawScoredRoi(image, roi, label=name, score=score)
                else:
                    raise Exception(
                        'Neither polygon, nor bbox is available for objectid %d'
                        % objectid)

        # Write an image.
        if args.image_path is not None:
            imagefile_new = imwriter.imwrite(image)
        else:
            imagefile_new = None

        # Write mask.
        if args.mask_path is not None and maskfile is not None:
            maskfile_new = imwriter.maskwrite(mask)
        else:
            maskfile_new = None

        # Update the database entry.
        if maskfile_new is not None:
            c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                      (maskfile_new, imagefile))
        if imagefile_new is not None:
            c.execute('UPDATE images SET imagefile=? WHERE imagefile=?',
                      (imagefile_new, imagefile))
            c.execute('UPDATE objects SET imagefile=? WHERE imagefile=?',
                      (imagefile_new, imagefile))

    imwriter.close()


def polygonsToMaskParser(subparsers):
    parser = subparsers.add_parser(
        'polygonsToMask',
        description=
        'Convert polygons of an object into a mask, and write it as maskfile.'
        'If there are polygon entries with different names, '
        'consider them as different polygons. '
        'Masks from each of these polygons are summed up and normalized '
        'to the number of these polygons. '
        'The result is a black-and-white mask when there is only one polygon, '
        'and a grayscale mask when there are multiple polygons.')
    parser.set_defaults(func=polygonsToMask)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='output either a directory with pictures or a video file')
    parser.add_argument(
        '--mask_path',
        help=
        'The directory for pictures OR video file, where masks are written to.'
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite images or video.')
    parser.add_argument('--skip_empty_masks',
                        action='store_true',
                        help='Do not write black masks with no objects.')
    parser.add_argument(
        '--substitute_with_box',
        action='store_true',
        help='if a polygon is not available, allow to use the bounding box.')


def polygonsToMask(c, args):
    imwriter = backendMedia.MediaWriter(media_type=args.media,
                                        rootdir=args.rootdir,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    # Iterate images.
    c.execute('SELECT imagefile,width,height FROM images')
    for imagefile, width, height in progressbar.progressbar(c.fetchall()):
        mask_per_image = np.zeros((height, width), dtype=np.int32)

        # Iterate objects.
        c.execute('SELECT objectid FROM objects WHERE imagefile=?',
                  (imagefile, ))
        for objectid, in c.fetchall():
            mask_per_object = np.zeros((height, width), dtype=np.int32)

            # Iterate multiple polygons (if any) of the object.
            c.execute('SELECT DISTINCT(name) FROM polygons WHERE objectid=?',
                      (objectid, ))
            polygon_names = c.fetchall()
            for polygon_name, in polygon_names:

                # Draw a polygon.
                if polygon_name is None:
                    c.execute('SELECT x,y FROM polygons WHERE objectid=?',
                              (objectid, ))
                else:
                    c.execute(
                        'SELECT x,y FROM polygons WHERE objectid=? AND name=?',
                        (objectid, polygon_name))
                pts = [[pt[0], pt[1]] for pt in c.fetchall()]
                logging.debug(
                    'Polygon "%s" of object %d consists of points: %s',
                    polygon_name, objectid, str(pts))
                mask_per_polygon = np.zeros((height, width), dtype=np.int32)
                cv2.fillPoly(mask_per_polygon,
                             [np.asarray(pts, dtype=np.int32)], 255)
                mask_per_object += mask_per_polygon

            # Area inside all polygons is white, outside all polygons is black, else gray.
            if len(polygon_names) > 1:
                mask_per_object = mask_per_object // len(polygon_names)

            # If there are no polygons, maybe substitute with roi.
            elif len(polygon_names) == 0 and args.substitute_with_box:
                c.execute('SELECT * FROM objects WHERE objectid=?',
                          (objectid, ))
                object_entry = c.fetchone()
                roi = backendDb.objectField(object_entry, 'roi')
                cv2.rectangle(mask_per_object, (roi[1], roi[0]),
                              (roi[3], roi[2]),
                              255,
                              thickness=-1)

            mask_per_image += mask_per_object
        mask_per_image = np.minimum(mask_per_image,
                                    255)  # Objects overlay on each other.
        mask_per_image = mask_per_image.astype(np.uint8)

        # Maybe skip empty mask.
        if np.sum(mask_per_image) == 0 and args.skip_empty_masks:
            continue

        out_maskfile = imwriter.maskwrite(mask_per_image,
                                          namehint=op.basename(
                                              op.splitext(imagefile)[0]))
        c.execute('UPDATE images SET maskfile=? WHERE imagefile=?',
                  (out_maskfile, imagefile))

    imwriter.close()


def writeMediaGridByTimeParser(subparsers):
    parser = subparsers.add_parser(
        'writeMediaGridByTime',
        description=
        'Export images, arranged in a grid, as a directory with pictures or as a video. '
        'Grid arranged by directory of imagefile (imagedir), and can be set manually.'
    )
    parser.set_defaults(func=writeMediaGridByTime)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='output either a directory with pictures or a video file.')
    parser.add_argument(
        '--image_path',
        required=True,
        help=
        'The directory for pictures OR video file, where images are written to.'
    )
    parser.add_argument('--winwidth',
                        type=int,
                        default=360,
                        help='target output width of each video in a grid.')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument(
        '--gridY',
        type=int,
        help='if specified, use the grid of "gridY" cells wide. Infer gridX.')
    parser.add_argument(
        '--imagedirs',
        nargs='+',
        help=
        'if specified, use these imagedirs instead of inferring from imagefile.'
    )
    parser.add_argument('--num_seconds',
                        type=int,
                        help='If specified, stop there.')
    parser.add_argument('--with_timestamp',
                        action='store_true',
                        help='Draw time on top.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite video if it exists.')


def writeMediaGridByTime(c, args):
    imreader = backendMedia.MediaReader(rootdir=args.rootdir)
    imwriter = backendMedia.MediaWriter(media_type=args.media,
                                        image_media=args.image_path,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    if not args.imagedirs:
        c.execute(
            "SELECT DISTINCT(rtrim(imagefile, replace(imagefile, '/', ''))) FROM images"
        )
        imagedirs = c.fetchall()
        imagedirs = [x.strip('/') for x, in imagedirs]
    else:
        imagedirs = args.imagedirs

    logging.info('Found %d distinct image directories/videos:\n%s',
                 len(imagedirs), pprint.pformat(imagedirs))
    num_cells = len(imagedirs)
    gridY = int(math.sqrt(num_cells)) if args.gridY is None else args.gridY
    gridX = num_cells // gridY
    logging.info('Will use grid %d x %d', gridY, gridX)

    # Will use the first image to find the target height from args.winwidth.
    # "width" and "height" are the target dimensions.
    c.execute('SELECT width,height FROM images')
    width, height = c.fetchone()
    height = height * args.winwidth // width
    width = args.winwidth

    c.execute('SELECT imagefile,timestamp FROM images')
    image_entries = c.fetchall()
    image_entries = [(imagefile, backendDb.parseTimeString(timestamp))
                     for imagefile, timestamp in image_entries]
    image_entries = sorted(image_entries, key=lambda x: x[1])

    time_min = min(image_entries, key=lambda x: x[1])[1]
    time_max = max(image_entries, key=lambda x: x[1])[1]
    logging.info('Min time: "%s", max time: "%s".', time_min, time_max)
    delta_seconds = (time_max - time_min).total_seconds()
    # User may limit the time to write.
    if args.num_seconds is not None:
        delta_seconds = min(delta_seconds, args.num_seconds)

    grid = None
    ientry = 0
    seconds_offsets = np.arange(0, int(delta_seconds), 1.0 / args.fps).tolist()

    for seconds_offset in progressbar.progressbar(seconds_offsets):

        out_time = time_min + datetime.timedelta(seconds=seconds_offset)
        logging.debug('Out-time=%s, in-time=%s', out_time,
                      image_entries[ientry][1])

        # For all the entries that happened before the frame.
        while image_entries[ientry][1] < out_time:

            imagefile, in_time = image_entries[ientry]
            logging.debug('Image entry %d.', ientry)
            ientry += 1
            # Skip those of no interest.
            if op.dirname(imagefile) not in imagedirs:
                logging.debug('Image dir %s not in %s', op.dirname(imagefile),
                              imagedirs)
                continue
            gridid = imagedirs.index(op.dirname(imagefile))

            # Read and scale.
            image = imreader.imread(imagefile)
            image = cv2.resize(image, dsize=(width, height))
            assert len(image.shape) == 3  # Now only color images.
            if args.with_timestamp:
                util.drawTextOnImage(image, backendDb.makeTimeString(in_time))

            # Lazy initialization.
            if grid is None:
                grid = np.zeros((height * gridY, width * gridX, 3),
                                dtype=np.uint8)

            logging.debug('Writing %s into gridid %d', imagefile, gridid)
            grid[height * (gridid // gridX):height * (gridid // gridX + 1),
                 width * (gridid % gridX):width *
                 (gridid % gridX + 1), :] = image.copy()

            imwriter.imwrite(grid)

    imwriter.close()


def repaintMaskParser(subparsers):
    parser = subparsers.add_parser(
        'repaintMask',
        description='Repaint specific colors in mask into different colors.')
    parser.set_defaults(func=repaintMask)
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=True,
        help='Output either a directory with pictures or a video file.')
    parser.add_argument(
        '--mask_path',
        help='the directory for pictures OR video file, where to write masks. '
        'Can not overwrite the same mask dir / video.')
    parser.add_argument(
        '--mask_mapping_dict',
        required=True,
        help=
        'values mapping for repainting. E.g. "{\'[1,254]\': [0,0,0], 255: [128,128,30]}"'
    )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite images or video.')
    parser.add_argument(
        '--out_rootdir',
        help='Specify, if rootdir changed for the output imagery.')


def repaintMask(c, args):
    labelmap = ast.literal_eval(args.mask_mapping_dict)
    logging.info('Parsed mask_mapping_dict to %s', pprint.pformat(labelmap))

    imreader = backendMedia.MediaReader(rootdir=args.rootdir)

    out_rootdir = args.out_rootdir if args.out_rootdir is not None else args.rootdir
    imwriter = backendMedia.MediaWriter(rootdir=out_rootdir,
                                        media_type=args.media,
                                        mask_media=args.mask_path,
                                        overwrite=args.overwrite)

    # Iterate images.
    c.execute('SELECT maskfile FROM images')
    maskfiles = c.fetchall()

    # Find out if all the masks are in the same directory.
    # If so, new maskfiles will have the same name as the original ones.
    # Otherwise, new maskfiles will be named sequentially from 0.
    # Regardless, namehint is currently only used when media=="pictures".
    maskdirs = set([op.dirname(maskfile) for maskfile in maskfiles])
    use_namehint = len(maskdirs) == 1

    for maskfile, in progressbar.progressbar(maskfiles):
        if maskfile is not None:
            # Read mask.
            mask = imreader.maskread(maskfile)
            # Repaint mask.
            mask = util.applyLabelMappingToMask(mask,
                                                labelmap).astype(np.uint8)
            # Write mask to video and to the db.
            maskfile_new = imwriter.maskwrite(
                mask, namehint=(maskfile if use_namehint else None))
            c.execute('UPDATE images SET maskfile=? WHERE maskfile=?',
                      (maskfile_new, maskfile))

    imwriter.close()
