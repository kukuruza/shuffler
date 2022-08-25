import os, os.path as op
import logging
import shutil
import progressbar

from lib.backend import backendDb
from lib.utils import util


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
        '--full_imagefile_as_name',
        action='store_true',
        help='If specified, imagefile entries will be made into the new file '
        'names by replacing "/" with "_". Otherwise, the last part of '
        'imagefile (imagename) will be used as new file names. '
        'Useful when files are from different dirs with duplicate names.')


def exportYolo(c, args):

    # Truncates numbers to N decimals
    def truncate(n, decimals=0):
        multiplier = 10**decimals
        return int(n * multiplier) / multiplier

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

        image_path = util.makeExportedImageName(image_dir, imagefile,
                                                args.full_imagefile_as_name)
        if args.copy_images:
            shutil.copyfile(src_image_path, image_path)
        elif args.symlink_images:
            os.symlink(op.abspath(src_image_path),
                       image_path,
                       target_is_directory=False)

        # Objects.
        lines = []
        c.execute(
            'SELECT name,x1,y1,width,height FROM objects WHERE imagefile=?',
            (imagefile, ))
        for name, x1, y1, width, height in c.fetchall():
            try:
                label_id = args.classes.index(name)
            except ValueError:
                continue

            xn = (x1 + width / 2.) / imwidth
            yn = (y1 + height / 2.) / imheight
            wn = width / imwidth
            hn = height / imheight

            line = f'{label_id} {truncate(xn, 7)} {truncate(yn, 7)} {truncate(wn, 7)} {truncate(hn, 7)}\n'
            logging.debug('label entry: %s', line)
            lines.append(line)

        if len(lines):
            labels_path = util.makeExportedImageName(
                labels_dir, imagefile, args.full_imagefile_as_name)
            labels_path = op.splitext(labels_path)[0] + '.txt'
            logging.debug('Writing labels to file: %s', labels_path)
            with open(labels_path, 'w') as f:
                f.writelines(lines)
        else:
            logging.debug('No labels in file, continue: %s', imagefile)
