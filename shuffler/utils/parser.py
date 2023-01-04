def addExportedImageNameArguments(parser):
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


def addKeepOrDeleteArguments(parser):
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
