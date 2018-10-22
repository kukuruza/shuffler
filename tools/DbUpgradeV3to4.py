#! /usr/bin/env python3
import sys, os, os.path as op
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import logging
import argparse
import progressbar
import sqlite3

from lib.util import copyWithBackup
from lib.backendDbUpgrade import upgradeV3toV4


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Upgrade schema from v3 to v4.')
  parser.add_argument('-i', '--in_db_path', required=True)
  parser.add_argument('-o', '--out_db_path')
  parser.add_argument('--no_backup', action='store_true', help='Do not backup')
  parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
    help='Log debug (10), info (20), warning (30), error (40).')
  args = parser.parse_args()

  progressbar.streams.wrap_stderr()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

  # Maybe back up.
  if args.out_db_path is not None:
    copyWithBackup(args.in_db_path, args.out_db_path)
    db_path = args.out_db_path
  else:
    db_path = args.in_db_path

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  upgradeV3toV4(cursor)
  
  conn.commit()
  conn.close()
