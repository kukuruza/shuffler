#! /usr/bin/env python3
import sys
import os.path as op
import logging
import argparse
import sqlite3
import progressbar
from itertools import groupby

from lib.utils import util
from lib.backend import backendDb
from lib import subcommands


def usage(name=None):
    return '''
    shuffler.py [-h] [-i INPUT.db] [-o OUTPUT.db] [--rootdir ROOTDIR]
      subcommand1  <subcommand's arguments>  '|'
      subcommand2  <subcommand's arguments>
    '''


def description():
    return '''
    Execute operations on a dataset.

    Create new or open an existing database, modify it with sub-commands,
    and optionally save the result as a different database.

    Available sub-commands are listed below as 'positional arguments'.
    Each sub-command has its own cmd arguments. Run a sub-command with -h flag
    to show its arguments. You can chain sub-commands with quoted vertical line '|'.

    More info and examples at https://github.com/kukuruza/shuffler.
    '''


def getParser():
    parser = argparse.ArgumentParser(
        prog='shuffler.py',
        usage=usage(),
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-i',
        '--in_db_file',
        required=False,
        help='If specified, open this file. Otherwise, create an empty db.')
    parser.add_argument(
        '-o',
        '--out_db_file',
        required=False,
        help='If specified, commit to this file. Otherwise, do not commit.')
    parser.add_argument(
        '--rootdir',
        default='.',
        help='If specified, images paths are relative to this dir.')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    subcommands.add_subparsers(parser)
    return parser


def connect(in_db_path=None, out_db_path=None):
    '''
    Connect to a new or existing database.
    Args:
      in_db_path:     If None, create a new database.
                      Otherwise, open a db at in_db_path.
      out_db_path:    If None, NOT commit (lose transactions).
                      Otherwise, back up out_db_path, if exists. Then commit.
                      Same logic applies when in_db_path==out_db_path.
    Returns:
      sqlite3.Connection
    '''

    if in_db_path is not None and not op.exists(in_db_path):
        raise FileNotFoundError('in_db_path specified but does not exist: %s' %
                                in_db_path)

    logging.info('in_db_path:  %s', in_db_path)
    logging.info('out_db_path: %s', out_db_path)

    if in_db_path is not None and not op.exists(in_db_path):
        raise IOError('Input database provided but does not exist: %s' %
                      in_db_path)

    if in_db_path is None and out_db_path is not None:
        # Create a new db and save as out_db_path.
        logging.info('will create database at %s', out_db_path)
        if op.exists(out_db_path):
            raise Exception(
                'Database "-o" exists. Specify it in "-i" too to change it.')
        conn = sqlite3.connect(out_db_path)
        backendDb.createDb(conn)

    elif in_db_path is not None and out_db_path is not None:
        # Load db from in_db_path and save as out_db_path.
        logging.info('will copy existing database from %s to %s.', in_db_path,
                     out_db_path)
        util.copyWithBackup(in_db_path, out_db_path)
        conn = sqlite3.connect(out_db_path)

    elif in_db_path is not None and out_db_path is None:
        # Load db from in_db_path as read-only.
        logging.info(
            'will load existing database from %s, but will not commit.',
            in_db_path)
        conn = backendDb.connect(in_db_path, 'load_to_memory')

    elif in_db_path is None and out_db_path is None:
        # Create a db and discard it in the end.
        logging.info('will create a temporary database in memory.')
        conn = sqlite3.connect(':memory:')  # Create an in-memory database.
        backendDb.createDb(conn)

    else:
        assert False

    return conn


def runSubcommand(cursor, args):
    print('=== %s ===' % args.func.__name__)
    args.func(cursor, args)


if __name__ == '__main__':
    parser = getParser()

    # If no argumernts were provided, add '-h' to print usage.
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    # Split command-line arguments into subcommands by special symbol "|".
    groups = groupby(sys.argv[1:], lambda x: x == '|')
    argv_splits = [list(group) for k, group in groups if not k]

    # Parse the main parser and the first subsparser.
    args = parser.parse_args(argv_splits[0])
    do_commit = args.out_db_file is not None
    rootdir = args.rootdir  # Copy, or it will get lost.

    # Logging was just parsed.
    progressbar.streams.wrap_stderr()
    FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'
    logging.basicConfig(level=args.logging, format=FORMAT)

    conn = connect(args.in_db_file, args.out_db_file)
    cursor = conn.cursor()

    if not hasattr(args, 'func'):
        raise ValueError(
            'Provide a sub-command. Type "./shuffler.py -h" for options.')

    # Main arguments and sub-command #1.
    runSubcommand(cursor, args)

    # Sub-commands #2 to last.
    for argv_split in argv_splits[1:]:
        args = parser.parse_args(argv_split)
        # "rootdir" is the only argument that is used in all subcommands.
        args.rootdir = rootdir
        runSubcommand(cursor, args)

    if do_commit:
        conn.commit()
        logging.info('Committed.')
    conn.close()
