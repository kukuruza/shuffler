import sys
import os.path as op
import logging
import argparse
import sqlite3
import progressbar
from itertools import groupby

from shuffler.utils import general as general_utils
from shuffler.backend import backend_db
from shuffler import operations


def usage():
    return '''
    shuffler [-h] [-i INPUT.db] [-o OUTPUT.db] [--rootdir ROOTDIR]
      operation1  <operation's arguments>  '|'
      operation2  <operation's arguments>
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
    operations.add_subparsers(parser)
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
        backend_db.createDb(conn)

    elif in_db_path is not None and out_db_path is not None:
        # Load db from in_db_path and save as out_db_path.
        logging.info('will copy existing database from %s to %s.', in_db_path,
                     out_db_path)
        general_utils.copyWithBackup(in_db_path, out_db_path)
        conn = sqlite3.connect(out_db_path)

    elif in_db_path is not None and out_db_path is None:
        # Load db from in_db_path as read-only.
        logging.info(
            'will load existing database from %s, but will not commit.',
            in_db_path)
        conn = backend_db.connect(in_db_path, 'load_to_memory')

    elif in_db_path is None and out_db_path is None:
        # Create a db and discard it in the end.
        logging.info('will create a temporary database in memory.')
        conn = sqlite3.connect(':memory:')  # Create an in-memory database.
        backend_db.createDb(conn)

    else:
        assert False

    return conn


def runOperation(cursor, args):
    print('=== %s ===' % args.func.__name__)
    logging.debug('=== Operation args: %s', args)
    args.func(cursor, args)


def main():
    parser = getParser()

    # If no argumernts were provided, add '-h' to print usage.
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    # Split command-line arguments into operations by special symbol "|".
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

    # in_db_file, out_db_file, and logging were aready used, remove them to
    # avoid the first function accidentally using them.
    del args.in_db_file
    del args.out_db_file
    del args.logging

    if not hasattr(args, 'func'):
        raise ValueError(
            'Provide a sub-command. Type "python -m shuffler -h" for options.')

    # Main arguments and sub-command #1.
    runOperation(cursor, args)

    # Sub-commands #2 to last.
    for argv_split in argv_splits[1:]:
        args = parser.parse_args(argv_split)
        # "rootdir" is the only argument that is used in all operations.
        args.rootdir = rootdir
        runOperation(cursor, args)

    if do_commit:
        conn.commit()
        logging.info('Committed.')
    conn.close()
