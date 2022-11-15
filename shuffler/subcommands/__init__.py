from shuffler.subcommands import dbGui, dbInfo, dbFilter, dbModify, dbMedia
from shuffler.subcommands import dbEvaluate, dbMatplotlib
from shuffler.subcommands.datasets import dbLabelme, dbKitti, dbPascal, dbBdd
from shuffler.subcommands.datasets import dbDetrac, dbCityscapes, dbCoco, dbYolo


def add_subparsers(parser):
    ''' Adds subparsers for each subcommand to the provided parser. '''
    subparsers = parser.add_subparsers()

    dbModify.add_parsers(subparsers)
    dbFilter.add_parsers(subparsers)
    dbGui.add_parsers(subparsers)
    dbInfo.add_parsers(subparsers)
    dbMedia.add_parsers(subparsers)
    dbEvaluate.add_parsers(subparsers)
    dbLabelme.add_parsers(subparsers)
    dbKitti.add_parsers(subparsers)
    dbPascal.add_parsers(subparsers)
    dbBdd.add_parsers(subparsers)
    dbDetrac.add_parsers(subparsers)
    dbCityscapes.add_parsers(subparsers)
    dbCoco.add_parsers(subparsers)
    dbYolo.add_parsers(subparsers)
    dbMatplotlib.add_parsers(subparsers)