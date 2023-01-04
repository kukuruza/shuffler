import os, os.path as op
import logging
import shutil
import progressbar

from shuffler.utils import general as general_utils
from shuffler.utils import parser as parser_utils


def add_parsers(subparsers):
    exportYoloParser(subparsers)


def exportYoloParser(subparsers):
    parser = subparsers.add_parser(
        'exportYolo',
        description='Export the database into YOLO format. '
        'Limitations: "segments" is TODO.')
    parser.set_defaults(func=exportYolo)
    parser.add_argument('--yolo_dir',
                        required=True,
                        help='Root directory of YOLO.')
    images_policy = parser.add_mutually_exclusive_group()
    images_policy.add_argument(
        '--copy_images',
        action='store_true',
        help=
        'If specified, will copy images to args.yolo_dir/"images"/args.subset. '
        'Required, if the media is stored as video. ')
    images_policy.add_argument(
        '--symlink_images',
        action='store_true',
        help='If specified, creates a symbolic link from imagefiles '
        'to imagefiles at args.coco_dir/"yolo"/args.subset/. '
        'Valid only if all the media is stored as images.')
    parser.add_argument('--subset',
                        required=True,
                        help='Name of the subset, such as "train2017')
    parser.add_argument(
        '--classes',
        required=True,
        help='Classes of interest in order. Will look at object names for them.'
    )
    parser.add_argument(
        '--as_polygons',
        action='store_true',
        help=
        'Save in the format of https://github.com/XinzeLee/PolygonObjectDetection'
    )
    parser_utils.addExportedImageNameArguments(parser)


# Truncates numbers to N decimals
def _truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def _exportImage(c, imagefile, imwidth, imheight, classes):
    lines = []
    c.execute('SELECT name,x1,y1,width,height FROM objects WHERE imagefile=?',
              (imagefile, ))
    for name, x1, y1, width, height in c.fetchall():
        try:
            label_id = classes.index(name)
        except ValueError:
            continue

        xn = (x1 + width / 2.) / imwidth
        yn = (y1 + height / 2.) / imheight
        wn = width / imwidth
        hn = height / imheight

        logging.debug('x1: %f, width: %f, imwidth: %f, xn: %f, wn: %f', x1,
                      width, imwidth, xn, wn)

        line = (f'{label_id} {_truncate(xn, 7)} {_truncate(yn, 7)} ' +
                f'{_truncate(wn, 7)} {_truncate(hn, 7)}\n')
        logging.debug('label entry: %s', line)
        lines.append(line)
    return lines


def _exportImageAsPolygons(c, imagefile, imwidth, imheight, classes):
    lines = []
    c.execute('SELECT objectid,name FROM objects WHERE imagefile=?',
              (imagefile, ))
    for objectid, name in c.fetchall():
        # In case bboxes were not recorded as polygons.
        general_utils.bboxes2polygons(c, objectid)

        try:
            label_id = classes.index(name)
        except ValueError:
            continue

        c.execute('SELECT x,y FROM polygons WHERE objectid=?', (objectid, ))
        polygon = c.fetchall()
        if len(polygon) != 4:
            logging.warning(
                'Polygon for objectid has %d points instead of 4. Skip.',
                len(polygon))
            continue

        x1 = polygon[0][0] / imwidth
        x2 = polygon[1][0] / imwidth
        x3 = polygon[2][0] / imwidth
        x4 = polygon[3][0] / imwidth
        y1 = polygon[0][1] / imheight
        y2 = polygon[1][1] / imheight
        y3 = polygon[2][1] / imheight
        y4 = polygon[3][1] / imheight

        line = (f'{label_id} ' + f'{_truncate(x1, 7)} {_truncate(y1, 7)} ' +
                f'{_truncate(x2, 7)} {_truncate(y2, 7)} ' +
                f'{_truncate(x3, 7)} {_truncate(y3, 7)} ' +
                f'{_truncate(x4, 7)} {_truncate(y4, 7)}\n')
        logging.debug('label entry: %s', line)
        lines.append(line)
    return lines


def exportYolo(c, args):

    # Images dir.
    image_dir = op.join(args.yolo_dir, 'images', args.subset)
    if op.exists(image_dir):
        shutil.rmtree(image_dir)
    if not op.exists(image_dir):
        os.makedirs(image_dir)

    # Labels dir.
    labels_dir = op.join(args.yolo_dir, 'labels', args.subset)
    if op.exists(labels_dir):
        shutil.rmtree(labels_dir)
    if not op.exists(labels_dir):
        os.makedirs(labels_dir)

    logging.info('Writing images.')
    c.execute('SELECT imagefile,width,height FROM images')
    for imagefile, imwidth, imheight in progressbar.progressbar(c.fetchall()):
        logging.debug('imagefile with: %s', imagefile)

        # Maybe copy or make a symlink for images.
        src_image_path = op.join(args.rootdir, imagefile)
        if not op.exists(src_image_path):
            raise FileNotFoundError(
                "Image not found at %s (using rootdir %s)." %
                (src_image_path, args.rootdir))

        image_path = general_utils.makeExportedImageName(
            image_dir, imagefile, args.dirtree_level_for_name,
            args.fix_invalid_image_names)
        if args.copy_images:
            shutil.copyfile(src_image_path, image_path)
        elif args.symlink_images:
            os.symlink(op.abspath(src_image_path),
                       image_path,
                       target_is_directory=False)

        # Objects.
        if args.as_polygons:
            lines = _exportImageAsPolygons(c, imagefile, imwidth, imheight,
                                           args.classes)
        else:
            lines = _exportImage(c, imagefile, imwidth, imheight, args.classes)

        # Write to labels file.
        if len(lines) > 0:
            labels_path = general_utils.makeExportedImageName(
                labels_dir, imagefile, args.dirtree_level_for_name,
                args.fix_invalid_image_names)
            labels_path = op.splitext(labels_path)[0] + '.txt'
            logging.debug('Writing labels to file: %s', labels_path)
            with open(labels_path, 'w') as f:
                f.writelines(lines)
        else:
            logging.debug('No labels in file, continue: %s', imagefile)
