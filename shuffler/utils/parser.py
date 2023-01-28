import argparse
from enum import Enum


class ArgumentType(Enum):
    NONE = 1
    OPTIONAL = 2
    REQUIRED = 3


def addExportedImageNameArguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--dirtree_level_for_name',
        type=int,
        default=1,
        help='How many levels of the directory structure to use as a filename. '
        'E.g. imagefile "my/fancy/image.jpg" would result in output name '
        '"image.jpg" when --dirtree_level_for_name=1, '
        '"fancy_image.jpg" when =2, and my_fancy_image.jpg with >=3. '
        'Useful when images in different dirs have the same filename.')
    parser.add_argument(
        '--fix_invalid_image_names',
        action='store_true',
        help='Replace invalid symbols with "_" in image names.')


def addKeepOrDeleteArguments(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--delete',
        dest='keep',
        action='store_false',
        help='Filtering will DELETE selected objects (the default behavior).')
    group.add_argument(
        '--keep',
        action='store_true',
        help='Filtering will KEEP selected objects, and DELETE the rest.')


def addMaskMappingArgument(parser: argparse.ArgumentParser):
    ''' Parser for using utis.general.applyMaskMapping. '''
    parser.add_argument('--mask_mapping',
                        default="\{\}",
                        help="""
A json to be parsed as a dict for remapping mask values.
Dict keys can be a combination of:
(a) scalar int in range [0, 255];
(b) strings of type ">N", "<N", ">=N", "<=N", or "[M,N]" are treated as a range;
Pixels absent from the mapping keys are left unchanged.

Dict values can be:
(a) scalars, which keeps the mask grayscale;
(b) tuple (r,g,b), which transforms the mask to color.

Examples of --mask_mapping:
(a) "{1: 255, 2: 128}"
    remaps 1 to 255, 2 to 128, and leaves the rest unchanged;
(b) "{1: '(255,255,255)', 2: '(128,255,0)'}"
    remaps 1 to (255,255,255), 2 to (128,255,0), and leaves the rest unchanged;
(c) "{'[0,254]': 0, '255': 1}
    remaps any pixel in range [0, 254] to 0 and 255 to 1.""")


def addWhereImageArgument(parser: argparse.ArgumentParser):
    ''' Adds a parser argument common for many operations. '''
    parser.add_argument(
        '--where_image',
        default='TRUE',
        help='an SQL "where" clause for the "images" table. '
        'E.g. to limit images to JPG pictures from directory "from/mydir", use '
        '\'images.imagefile LIKE "from/mydir/%%.JPG"\'')


def addWhereObjectArgument(parser: argparse.ArgumentParser,
                           name: str = '--where_object'):
    ''' Adds a parser argument common for many operations. '''
    parser.add_argument(name,
                        default='TRUE',
                        help='an SQL "where" clause for the "objects" table. '
                        'E.g. to limit objects to those named "car", use '
                        '\'objects.name == "car"\'')


def addMediaOutputArguments(parser: argparse.ArgumentParser,
                            image_path: ArgumentType = ArgumentType.NONE,
                            mask_path: ArgumentType = ArgumentType.NONE,
                            out_rootdir: bool = False):
    ''' 
    Arguments to save images and/or masks to media (pictures or video). 
    These are all the arguments needed for backend_media.MediaWriter.__init__().
    '''

    assert image_path != ArgumentType.NONE or mask_path != ArgumentType.NONE

    # --image_path is added, if needed.
    if image_path != ArgumentType.NONE:
        parser.add_argument(
            '--image_path',
            required=(image_path == ArgumentType.REQUIRED),
            help=
            'The directory for pictures OR video file, where images are written to.'
        )

    # --mask_path is added, if needed.
    if mask_path != ArgumentType.NONE:
        parser.add_argument(
            '--mask_path',
            required=(mask_path == ArgumentType.REQUIRED),
            help=
            'The directory for pictures OR video file, where masks are written to.'
        )

    # --media is always added.
    # It is required if --image_path or --mask_path are required.
    parser.add_argument(
        '--media',
        choices=['pictures', 'video'],
        required=(image_path == ArgumentType.REQUIRED
                  or mask_path == ArgumentType.REQUIRED),
        help='Output either a directory with pictures or a video file')

    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite result if it exists.')

    # Used if user writes media as a by-product, e.g. for visualization.
    if out_rootdir:
        parser.add_argument(
            '--out_rootdir',
            help='Specify, if rootdir changed for the output imagery.')
