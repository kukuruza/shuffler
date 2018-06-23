import os, sys, os.path as op
import numpy as np
import cv2
import logging
from glob import glob
from pprint import pformat
from backendImages import getVideoLength

def add_parsers(subparsers):
  filterWithAnotherParser(subparsers)


def filterWithAnotherParser(subparsers):
  parser = subparsers.add_parser('filterWithAnother',
      description='Remove images from the db that are not in the reference db.')
  parser.add_argument('--ref_db_file', required=True)
  parser.set_defaults(func=filterWithAnother)

def filterWithAnother (c, c_ref):
  logging.info ('==== filterWithAnother ====')

  # Get all the imagefiles from the reference db.
  conn_ref = loadToMemory(args.ref_db_file)
  c_ref = conn_ref.cursor()
  c_ref.execute('SELECT imagefile FROM images')
  imagefiles_ref = c_ref.fetchall()
  imagenames_ref = [op.basename(x) for x, in imagefiles_ref]
  logging.info ('Total %d images in ref.' % len(imagefiles_ref))
  conn_ref.close()

  # Get all the imagefiles from the main db.
  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()

  # Delete.
  imagefiles_del = [x for x, in imagefiles if op.basename(x) not in imagenames_ref]
  for imagefile_del in imagefiles_del:
    c.execute('DELETE FROM images WHERE imagefile=?', (imagefile_del,))
    c.execute('DELETE FROM cars   WHERE imagefile=?', (imagefile_del,))
