#! /usr/bin/env python3
import sys
import os.path as op
import logging
import argparse
import sqlite3
import progressbar
from itertools import groupby

from lib.util import copyWithBackup
from lib.backendDb import createDb
from lib import dbGui, dbInfo, dbFilter, dbModify, dbWrite, dbEvaluate
from lib import dbLabelme, dbKitti, dbPascal


def connect (in_db_path=None, out_db_path=None):
  ''' Connect to a new or existing database.
  Args:
    in_db_path:   If None, create a new database.
                  Otherwise, open a db at in_db_path.
    out_db_path:  If None, NOT commit (lose transactions).
                  Otherwise, back up out_db_path, if exists. Then commit.
                  Same logic applies when in_db_path==out_db_path.
  Returns:
    sqlite3.Connection
  '''

  logging.info('in_db_path:  %s' % in_db_path)
  logging.info('out_db_path: %s' % out_db_path)

  if in_db_path is not None and not op.exists(in_db_path):
    raise IOError('Input database provided but does not exist: %s' % in_db_path)

  if in_db_path is None and out_db_path is not None:
    # Create a new db and save as out_db_path.
    logging.info('will create database at %s' % out_db_path)
    if op.exists(out_db_path):
      raise Exception('Database "-o" exists. Specify it in "-i" too to change it.')
    conn = sqlite3.connect(out_db_path)
    createDb(conn)

  elif in_db_path is not None and out_db_path is not None:
    # Load db from in_db_path and save as out_db_path.
    logging.info('will copy existing database from %s to %s.' % (in_db_path, out_db_path))
    copyWithBackup(in_db_path, out_db_path)
    conn = sqlite3.connect(out_db_path)

  elif in_db_path is not None and out_db_path is None:
    # Load db from in_db_path as read-only.
    logging.info('will load existing database from %s, but will not commit.' % in_db_path)
    conn_in = sqlite3.connect(in_db_path)
    conn = sqlite3.connect(':memory:')  # Create a memory database.
    query = ''.join(line for line in conn_in.iterdump())
    conn.executescript(query)  # Dump input database in the one in memory.
    # Alternative implementation is commented out below,
    #   but in that iomplementation one can't modify the database
    #   even without committing it in the end.
    #conn = sqlite3.connect('file:%s?mode=ro' % in_db_path, uri=True)

  elif in_db_path is None and out_db_path is None:
    # Create a db and discard it in the end.
    logging.info('will create a temporary database in memory.')
    conn = sqlite3.connect(':memory:')  # Create an in-memory database.
    createDb(conn)

  else:
    assert False

  return conn


parser = argparse.ArgumentParser(description=
  '''Create new or open existing database, modify it with sub-commands,
  and optionally save the result in another database.
  Positional arguments correspond to sub-commands.
  Each tool has its own arguments, run a tool with -h flag
  to show its arguments. You can use chain sub-commands.
  ''')
parser.add_argument('-i', '--in_db_file', required=False,
    help='If specified, open this file. Otherwise, create an empty db.')
parser.add_argument('-o', '--out_db_file', required=False,
    help='If specified, commit to this file. Otherwise, do not commit.')
parser.add_argument('--rootdir', default='.',
    help='If specified, images paths are relative to this dir.')
parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
subparsers = parser.add_subparsers()
dbModify.add_parsers(subparsers)
dbFilter.add_parsers(subparsers)
dbGui.add_parsers(subparsers)
dbInfo.add_parsers(subparsers)
dbWrite.add_parsers(subparsers)
dbEvaluate.add_parsers(subparsers)
dbLabelme.add_parsers(subparsers)
dbKitti.add_parsers(subparsers)
dbPascal.add_parsers(subparsers)

# Split command-line arguments into subcommands by special symbol "|".
argv_splits = [list(group) for k, group in groupby(sys.argv[1:], lambda x: x == '|') if not k]

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

# Go thourgh the pipeline.
print(args.func.__name__)
args.func(cursor, args)
for argv_split in argv_splits[1:]:
  args = parser.parse_args(argv_split)
  args.rootdir = rootdir  # This is the only argument that is used in all subcommands.
  print(args.func.__name__)
  args.func(cursor, args)

if do_commit:
  conn.commit()
  logging.info('Committed.')
conn.close()
