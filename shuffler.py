#! /usr/bin/env python3
import os.path as op
import logging
import argparse
import sqlite3
import progressbar
from lib.utilities import safeCopy
from lib.backendDb import create, loadToMemory
from lib import dbModify, dbVideo, dbInfo, dbDisplay, dbFilter, dbEvaluate
#import dbExport, dbLabel, dbLabelme


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

  logging.info ('in_db_path:  %s' % in_db_path)
  logging.info ('out_db_path: %s' % out_db_path)

  if in_db_path is None and out_db_path is not None:
    logging.info('will create database at %s' % out_db_path)
    if op.exists(out_db_path):
      raise Exception('Database "-o" exists. Specify it in "-i" too to change it.')
    conn = sqlite3.connect(out_db_path)
    create(conn)

  elif in_db_path is not None and out_db_path is not None:
    logging.info('will copy existing database from %s to %s.' % (in_db_path, out_db_path))
    safeCopy(in_db_path, out_db_path)
    conn = sqlite3.connect(out_db_path)

  elif in_db_path is not None and out_db_path is None:
    logging.info('will load existing database from %s, but will not commit).' % in_db_path)
    conn = sqlite3.connect('file:%s?mode=ro' % in_db_path, uri=True)

  elif in_db_path is None and out_db_path is None:
    logging.info('will create a temporary database in memory.')
    conn = sqlite3.connect(':memory:')  # Create an in-memory database.
    create(conn)

  else:
    assert False

  return sqlite3.Connection


parser = argparse.ArgumentParser(description=
  '''Create new or open existing database, modify it with sub-commands,
  and optionally save the result in another database.
  Positional arguments correspond to sub-commands.
  Each tool has its own arguments, run a tool with -h flag
  to show its arguments. You can use chain sub-commands.
  ''')
parser.add_argument('-i', '--in_db_file', required=False,
    help='If specified, open this file. If unspecified create out_db_file')
parser.add_argument('-o', '--out_db_file', required=False,
    help='If specified, write to that file. If unspecified, do not commit')
parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
subparsers = parser.add_subparsers()
dbVideo.add_parsers(subparsers)
dbModify.add_parsers(subparsers)
dbDisplay.add_parsers(subparsers)
dbInfo.add_parsers(subparsers)
# dbExport.add_parsers(subparsers)
# dbCadFilter.add_parsers(subparsers)
# dbLabel.add_parsers(subparsers)
dbEvaluate.add_parsers(subparsers)
dbFilter.add_parsers(subparsers)
# dbLabelme.add_parsers(subparsers)
# Add a dummy option to allow passing '--' in order to end lists.
dummy = subparsers.add_parser('--')
dummy.set_defaults(func=lambda *args: None)


# Parse main parser and the first subsparser.
args, rest = parser.parse_known_args()
out_db_file = args.out_db_file  # Copy, or it will get lost.

progressbar.streams.wrap_stderr()
FORMAT = '[%(filename)s:%(lineno)s - %(funcName)20s() %(levelname)s]: %(message)s'
logging.basicConfig(level=args.logging, format=FORMAT)

conn = dbInit(args.in_db_file, args.out_db_file)

# Go thourgh the pipeline.
args.func(conn.cursor(), args)
while rest:
  args, rest = parser.parse_known_args(rest)
  args.func(conn.cursor(), args)

if out_db_file is not None:
  conn.commit()
  logging.info('Commited.')
conn.close()

