import os, sys, os.path as op
import argparse
import numpy as np
import logging
from pprint import pprint
from itertools import groupby

from .backendDb import imageField


def add_parsers(subparsers):
  plotObjectsHistogramParser(subparsers)
  plotObjectsScatterParser(subparsers)
  plotObjectsStripParser(subparsers)
  printInfoParser(subparsers)
  dumpDbParser(subparsers)


def _maybeNumerizeProperty(values):
  ''' Field "value" of table "properties" is string. Maybe the data is float. '''
  try:
    return [float(value) for value in values]
  except ValueError:
    return values


def plotObjectsHistogramParser(subparsers):
  parser = subparsers.add_parser('plotObjectsHistogram',
    description='Get a 1d histogram plot of fields of objects.')
  parser.set_defaults(func=plotObjectsHistogram)
  parser.add_argument('--sql_query',
    help='SQL query for ONE field, the "x" in the plot. '
    'Example: \'SELECT value FROM properties WHERE key="pitch"\'')
  parser.add_argument('--xlabel', required=True)
  parser.add_argument('--ylog', action='store_true')
  parser.add_argument('--bins', type=int)
  parser.add_argument('--categorical', action='store_true')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotObjectsHistogram(c, args):
  import matplotlib.pyplot as plt

  c.execute(args.sql_query)
  object_entries = c.fetchall()
  
  # Clean data.
  if not object_entries:
    logging.info('No objects, nothing to draw.')
    return
  if len(object_entries[0]) != 1:
    raise ValueError('Must query for 1 fields, not %d.' % len(object_entries[0]))
  xlist = [x for x, in object_entries if x is not None]
  logging.info('%d objects have a non-None field.' % len(xlist))

  # List to proper format.
  xlist = _maybeNumerizeProperty(xlist)
  logging.debug('%s' % (str(xlist)))

  fig, ax = plt.subplots()
  if args.categorical:
    import pandas as pd
    import seaborn as sns
    data = pd.DataFrame({args.x: xlist})
    ax = sns.countplot(x="name", data=data, order=data[args.x].value_counts().index)
  else:
    if args.bins:
      ax.hist(xlist, args.bins)
    else:
      ax.hist(xlist)

  if args.ylog:
    ax.set_yscale('log', nonposy='clip')
  plt.xlabel(args.xlabel)
  plt.ylabel('')
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)
  if args.display:
    plt.show()


def plotObjectsStripParser(subparsers):
  parser = subparsers.add_parser('plotObjectsStrip',
    description='Get a 1d histogram plot of fields of objects.')
  parser.set_defaults(func=plotObjectsStrip)
  parser.add_argument('--sql_query',
    help='SQL query for TWO fields, the "x" and the "y" in the plot. '
    'Example: \'SELECT p1.value, p2.value FROM properties p1 '
    'INNER JOIN properties p2 ON p1.objectid=p2.objectid '
    'WHERE p1.key="pitch" AND p2.key="yaw"\'')
#    'Example: \'SELECT name, p1.value, p2.value FROM objects '
#    'INNER JOIN properties p1 ON objects.objectid=p1.objectid INNER JOIN properties p2 '
#    'ON p1.objectid=p2.objectid WHERE name="bus" AND p1.key="pitch" AND p2.key="yaw"\'')
  parser.add_argument('--xlabel', required=True)
  parser.add_argument('--ylabel', required=True)
  parser.add_argument('--jitter', action='store_true')
  parser.add_argument('--ylog', action='store_true')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotObjectsStrip(c, args):
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns

  c.execute(args.sql_query)
  object_entries = c.fetchall()
  
  # Clean data.
  if not object_entries:
    logging.info('No objects, nothing to draw.')
    return
  if len(object_entries[0]) != 2:
    raise ValueError('Must query for 2 fields, not %d.' % len(object_entries[0]))
  xylist = [(x,y) for (x,y) in object_entries if x is not None and y is not None]
  logging.info('%d objects have both fields non-None.' % len(xylist))

  # From a list of tuples to two lists.
  xlist, ylist = tuple(map(list, zip(*xylist)))
  xlist = _maybeNumerizeProperty(xlist)
  ylist = _maybeNumerizeProperty(ylist)
  logging.debug('%s\n%s' % (str(xlist), str(ylist)))

  fig, ax = plt.subplots()
  data = pd.DataFrame({args.xlabel: xlist, args.ylabel: ylist})
  ax = sns.stripplot(x=args.xlabel, y=args.ylabel, data=data, jitter=args.jitter)

  if args.ylog:
    ax.set_yscale('log', nonposy='clip')
  plt.xlabel(args.xlabel)
  plt.ylabel(args.ylabel)
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)
  if args.display:
    plt.show()


def plotObjectsScatterParser(subparsers):
  parser = subparsers.add_parser('plotObjectsScatter',
    description='Get a 2d scatter plot of fields of objects.')
  parser.set_defaults(func=plotObjectsScatter)
  parser.add_argument('--sql_query',
    help='SQL query for TWO fields, the "x" and the "y" in the plot. '
    'Example: \'SELECT p1.value, p2.value FROM properties p1 '
    'INNER JOIN properties p2 ON p1.objectid=p2.objectid '
    'WHERE p1.key="pitch" AND p2.key="yaw"\'')
  parser.add_argument('--xlabel', required=True)
  parser.add_argument('--ylabel', required=True)
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotObjectsScatter(c, args):
  import matplotlib.pyplot as plt

  c.execute(args.sql_query)
  object_entries = c.fetchall()
  
  # Clean data.
  if not object_entries:
    logging.info('No objects, nothing to draw.')
    return
  if len(object_entries[0]) != 2:
    raise ValueError('Must query for 2 fields, not %d.' % len(object_entries[0]))
  xylist = [(x,y) for (x,y) in object_entries if x is not None and y is not None]
  logging.info('%d objects have both fields non-None.' % len(xylist))

  # From a list of tuples to two lists.
  xlist, ylist = tuple(map(list, zip(*xylist)))
  xlist = _maybeNumerizeProperty(xlist)
  ylist = _maybeNumerizeProperty(ylist)
  logging.debug('%s\n%s' % (str(xlist), str(ylist)))

  plt.scatter(xlist, ylist, alpha=0.5)
  plt.xlabel(args.xlabel)
  plt.ylabel(args.ylabel)
  if args.display:
    plt.show()
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)


def _getImagesStats(image_entries):
  ''' Process image entries into stats. '''
  info = {}
  info['num images'] = len(image_entries)
  info['num masks'] = len([x for x in image_entries if imageField(x, 'maskfile') is not None])
  # Width.
  widths = set([imageField(x, 'width') for x in image_entries])
  if len(widths) > 1:
    info['image width'] = '%s different values' % len(widths)
  elif len(widths) == 1:
    info['image width'] = '%s' % next(iter(widths))
  # Height.
  heights = set([imageField(x, 'height') for x in image_entries])
  if len(heights) > 1:
    info['image height'] = '%s different values' % len(heights)
  elif len(heights) == 1:
    info['image height'] = '%s' % next(iter(heights))
  return info

def _getObjectsStats(object_entries):
  ''' Process object entries into stats. '''
  info = {}
  info['num objects'] = len(object_entries)
  return info

def printInfoParser(subparsers):
  parser = subparsers.add_parser('printInfo',
    description='Show major info of database.')
  parser.set_defaults(func=printInfo)
  parser.add_argument('--images_by_dir', action='store_true',
    help='Print image statistics by directory.')
  parser.add_argument('--objects_by_image', action='store_true',
    help='Print object statistics by directory.')

def printInfo (c, args):
  info = {}

  # Images stats.
  c.execute('SELECT * FROM images')
  image_entries = c.fetchall()
  if args.images_by_dir:
    # Split by image directories.
    for key, group in groupby(image_entries, lambda x: op.dirname(imageField(x, 'imagefile'))):
      info['imagedir="%s"' % key] = _getImagesStats(list(group))
  else:
    info.update(_getImagesStats(image_entries))

  # Objects stats.
  c.execute('SELECT * FROM objects')
  object_entries = c.fetchall()
  if args.objects_by_image:
    # Split by image directories.
    for key, group in groupby(object_entries, lambda x: imageField(x, 'imagefile')):
      info['image="%s"' % key] = _getObjectsStats(list(group))
  else:
    info.update(_getObjectsStats(object_entries))

  # Properties (basic for now.)
  c.execute('SELECT DISTINCT(key) FROM properties')
  property_names = c.fetchall()
  info['properties'] = [x for x, in property_names]

  # Matches stats (basic for now.)
  c.execute('SELECT DISTINCT(match) FROM matches')
  matches = c.fetchall()
  info['matches'] = len(matches)

  pprint (info)


def dumpDbParser(subparsers):
  parser = subparsers.add_parser('dumpDb',
    description='Print tables of the database.')
  parser.add_argument('--tables', nargs='+',
    choices=['images', 'objects', 'properties', 'polygons', 'matches'],
    default=['images', 'objects', 'properties', 'polygons', 'matches'],
    help='Tables to print out, all by default.')
  parser.add_argument('--limit', type=int,
    help='How many entries.')
  parser.set_defaults(func=dumpDb)

def dumpDb (c, args):
  
  def _dumpTable(tablename):
    print('Table: "%s":' % tablename)
    if args.limit:
      c.execute('SELECT * FROM %s LIMIT %d' % (tablename, args.limit))
    else:
      c.execute('SELECT * FROM %s' % tablename)
    for entry in c.fetchall():
      print(entry)

  for table in args.tables:
    _dumpTable(table)

