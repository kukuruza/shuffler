from lib.subcommands import dbGui, dbInfo, dbFilter, dbModify, dbMedia
from lib.subcommands import dbEvaluate, dbLabel, dbMatplotlib
from lib.subcommands.datasets import dbLabelme, dbKitti, dbPascal, dbBdd
from lib.subcommands.datasets import dbDetrac, dbCityscapes, dbCoco


def add_subparsers(parser):
    ''' Adds subparsers for each subcommand to the provided parser. '''
    subparsers = parser.add_subparsers()

    dbModify.add_parsers(subparsers)
    dbFilter.add_parsers(subparsers)
    dbGui.add_parsers(subparsers)
    dbInfo.add_parsers(subparsers)
    dbMedia.add_parsers(subparsers)
    dbEvaluate.add_parsers(subparsers)
    dbLabel.add_parsers(subparsers)
    dbLabelme.add_parsers(subparsers)
    dbKitti.add_parsers(subparsers)
    dbPascal.add_parsers(subparsers)
    dbBdd.add_parsers(subparsers)
    dbDetrac.add_parsers(subparsers)
    dbCityscapes.add_parsers(subparsers)
    dbCoco.add_parsers(subparsers)
    dbMatplotlib.add_parsers(subparsers)
