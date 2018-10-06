import os, sys, os.path as op
import argparse
import numpy as np
import logging
from pprint import pprint
from .backendDb import carField


def add_parsers(subparsers):
  plotHistogramParser(subparsers)
  plotScatterParser(subparsers)
  plotStripParser(subparsers)
  infoParser(subparsers)


def plotHistogramParser(subparsers):
  parser = subparsers.add_parser('plotHistogram',
    description='Get a 1d histogram plot of fields of cars.')
  parser.set_defaults(func=plotHistogram)
  parser.add_argument('-x', required=True)
  parser.add_argument('--ylog', action='store_true')
  parser.add_argument('--bins', type=int)
  parser.add_argument('--xlabel', action='store_true')
  parser.add_argument('--categorical', action='store_true')
  parser.add_argument('--constraint', default='1')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotHistogram(c, args):
  logging.info ('==== plotHistogram ====')
  import matplotlib.pyplot as plt

  c.execute('SELECT %s FROM cars WHERE %s' % (args.x, args.constraint))
  car_entries = c.fetchall()

  xlist = [x for x, in car_entries if x is not None]
  logging.info('Out of %d cars, %d are valid.' % (len(car_entries), len(xlist)))
  if not xlist:
    logging.info('No cars, nothing to draw.')
    return
  logging.debug(str(xlist))

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
  plt.xlabel(args.x if args.xlabel else '')
  plt.ylabel('')
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)
  if args.display:
    plt.show()


def plotStripParser(subparsers):
  parser = subparsers.add_parser('plotStrip',
    description='Get a 1d histogram plot of fields of cars.')
  parser.set_defaults(func=plotStrip)
  parser.add_argument('-x', required=True)
  parser.add_argument('-y', required=True)
  parser.add_argument('--xlabel', action='store_true')
  parser.add_argument('--ylabel', action='store_true')
  parser.add_argument('--jitter', action='store_true')
  parser.add_argument('--ylog', action='store_true')
  parser.add_argument('--constraint', default='1')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotStrip(c, args):
  logging.info ('==== plotStrip ====')
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns

  c.execute('SELECT %s,%s FROM cars WHERE %s' % (args.x, args.y, args.constraint))
  car_entries = c.fetchall()

  xylist = [(x,y) for (x,y) in car_entries if x is not None and y is not None]
  logging.info('Out of %d cars, %d are valid.' % (len(car_entries), len(xylist)))
  if not xylist:
    logging.info('No cars, nothing to draw.')
    return
  xlist, ylist = tuple(map(list, zip(*xylist)))
  logging.debug(str(xlist) + '\n' + str(ylist))

  fig, ax = plt.subplots()
  data = pd.DataFrame({args.x: xlist, args.y: ylist})
  ax = sns.stripplot(x=args.x, y=args.y, data=data, jitter=args.jitter)

  if args.ylog:
    ax.set_yscale('log', nonposy='clip')
  plt.xlabel(args.x if args.xlabel else '')
  plt.ylabel(args.y if args.ylabel else '')
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)
  if args.display:
    plt.show()


def plotScatterParser(subparsers):
  parser = subparsers.add_parser('plotScatter',
    description='Get a 2d scatter plot of fields of cars.')
  parser.set_defaults(func=plotScatter)
  parser.add_argument('-x', required=True)
  parser.add_argument('-y', required=True)
  parser.add_argument('--constraint', default='1')
  parser.add_argument('--display', action='store_true')
  parser.add_argument('--out_path')

def plotScatter(c, args):
  logging.info ('==== plotScatter ====')
  import matplotlib.pyplot as plt

  c.execute('SELECT %s,%s FROM cars WHERE %s' % (args.x, args.y, args.constraint))
  car_entries = c.fetchall()

  xylist = [(x,y) for (x,y) in car_entries if x is not None and y is not None]
  logging.info('Out of %d cars, %d are valid.' % (len(car_entries), len(xylist)))
  if not xylist:
    logging.info('No cars, nothing to draw.')
    return
  xlist, ylist = tuple(map(list, zip(*xylist)))
  logging.debug(str(xlist) + '\n' + str(ylist))

  plt.scatter(xlist, ylist, alpha=0.5)
  plt.xlabel(args.x)
  plt.ylabel(args.y)
  if args.display:
    plt.show()
  if args.out_path:
    logging.info('Saving to %s' % args.out_path)
    plt.savefig(args.out_path)


def infoParser(subparsers):
  parser = subparsers.add_parser('info',
    description='Show major info of database.')
  parser.set_defaults(func=info)
  parser.add_argument('--imagedirs', action='store_true')
  parser.add_argument('--imagerange', action='store_true')

def info (c, args):
  logging.info ('==== info ====')
  info = {}

  c.execute('SELECT imagefile FROM images')
  imagefiles = c.fetchall()
  info['numimages'] = len(imagefiles)
  imagedirs = [op.dirname(x) for x, in imagefiles]
  imagedirs = list(set(imagedirs))
  if args.imagedirs:
    info['imagedirs'] = imagedirs
    for i,imagedir in enumerate(imagedirs):
      #imagefile = [x for x in imagefiles if imagedir in x][0]
      imagefile = imagefiles[0]
      c.execute('SELECT width,height FROM images WHERE imagefile=?', imagefile)
      (width, height) = c.fetchone()
      print (width, height)

  def collectImageRange(nums):
    if len(nums) == 0: return []
    nums = sorted(nums)
    numrange = []
    start = current = nums[0]
    for num in nums[1:]:
      if num > current + 1:
        numrange.append((start, current+1))
        start = num
        if len(numrange) >= 2:
          numrange.append('too many ranges')
          break
      current = num
    return numrange
  if args.imagerange:
    info['imagerange'] = {}
    for imagedir in imagedirs:
      imagenums = [int(op.basename(x)) for x, in imagefiles if op.dirname(x) == imagedir]
      info['imagerange'][imagedir] = collectImageRange(imagenums)

  #c.execute('SELECT maskfile FROM images')
  #maskfiles = c.fetchall()
  #maskdirs = [op.dirname(x) for x, in maskfiles]
  #maskdirs = list(set(maskdirs))
  #info['maskdirs'] = maskdirs

  c.execute('SELECT COUNT(*) FROM cars')
  info['numcars'] = c.fetchone()[0]
  pprint (info, width=120)


