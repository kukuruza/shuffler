from shuffler.operations import evaluate, filtering, gui, info, media, modify
from shuffler.operations import matplotlib
from shuffler.operations.datasets import bdd, cityscapes, coco, pascal, detrac, kitti, labelme, yolo


def add_subparsers(parser):
    ''' Adds subparsers for each operation to the provided parser. '''
    subparsers = parser.add_subparsers()

    modify.add_parsers(subparsers)
    filtering.add_parsers(subparsers)
    gui.add_parsers(subparsers)
    info.add_parsers(subparsers)
    media.add_parsers(subparsers)
    evaluate.add_parsers(subparsers)
    labelme.add_parsers(subparsers)
    kitti.add_parsers(subparsers)
    pascal.add_parsers(subparsers)
    bdd.add_parsers(subparsers)
    detrac.add_parsers(subparsers)
    cityscapes.add_parsers(subparsers)
    coco.add_parsers(subparsers)
    yolo.add_parsers(subparsers)
    matplotlib.add_parsers(subparsers)
