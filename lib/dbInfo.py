import os, sys, os.path as op
import argparse
import numpy as np
import logging
from pprint import pprint
from itertools import groupby

from .backendDb import imageField


def add_parsers(subparsers):
    plotHistogramParser(subparsers)
    plotScatterParser(subparsers)
    plotStripParser(subparsers)
    plotViolinParser(subparsers)
    printInfoParser(subparsers)
    dumpDbParser(subparsers)


def _maybeNumerizeProperty(values):
    '''
    Field "value" of table "properties" is string. Maybe the data is float.
    '''
    try:
        return [float(value) for value in values]
    except ValueError:
        return values


def plotHistogramParser(subparsers):
    parser = subparsers.add_parser(
        'plotHistogram',
        description='Get a 1d histogram plot of a field in the db.')
    parser.set_defaults(func=plotHistogram)
    parser.add_argument(
        '--sql',
        help='SQL query for ONE field, the "x" in the plot. '
        'Example: \'SELECT value FROM properties WHERE key="pitch"\'')
    parser.add_argument('--xlabel')
    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--bins', type=int)
    parser.add_argument('--xlim', type=float, nargs='+')
    parser.add_argument('--rotate_xlabels',
                        type=float,
                        default=0.,
                        help='Rotate labels of x-axis ticks.')
    parser.add_argument('--categorical', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotHistogram(c, args):
    import matplotlib.pyplot as plt

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if len(entries[0]) != 1:
        raise ValueError('Must query for 1 fields, not %d.' % len(entries[0]))
    xlist = [x for x, in entries if x is not None]
    logging.info('%d entries have a non-None field.' % len(xlist))

    # List to proper format.
    xlist = _maybeNumerizeProperty(xlist)
    logging.debug('%s' % (str(xlist)))

    fig, ax = plt.subplots()
    if args.categorical:
        import pandas as pd
        import seaborn as sns
        data = pd.DataFrame({args.xlabel: xlist})
        ax = sns.countplot(x=args.xlabel,
                           data=data,
                           order=data[args.xlabel].value_counts().index)
    else:
        ax.hist(xlist, bins=args.bins)
    plt.xticks(rotation=args.rotate_xlabels)
    #fig.subplots_adjust(bottom=0.25)

    if args.xlim:
        if not len(args.xlim) == 2:
            raise Exception('Argument xlim requires to numbers')
        plt.xlim(args.xlim)
    if args.ylog:
        ax.set_yscale('log', nonposy='clip')
    if args.xlabel:
        plt.xlabel(args.xlabel)
    plt.ylabel('')
    if args.out_path:
        logging.info('Saving to %s' % args.out_path)
        plt.savefig(args.out_path)
    if args.display:
        plt.show()


def plotStripParser(subparsers):
    parser = subparsers.add_parser(
        'plotStrip', description='Get a "strip" plot of two fields in the db.')
    parser.set_defaults(func=plotStrip)
    parser.add_argument(
        '--sql',
        help='SQL query for TWO fields, the "x" and the "y" in the plot. '
        'Example: \'SELECT p1.value, p2.value FROM properties p1 '
        'INNER JOIN properties p2 ON p1.objectid=p2.objectid '
        'WHERE p1.key="pitch" AND p2.key="yaw"\'')
    parser.add_argument('--xlabel', required=True)
    parser.add_argument('--ylabel', required=True)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--rotate_xlabels',
                        type=float,
                        default=0.,
                        help='Rotate labels of x-axis ticks.')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotStrip(c, args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.' % len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s' % (str(xlist), str(ylist)))

    fig, ax = plt.subplots()
    data = pd.DataFrame({args.xlabel: xlist, args.ylabel: ylist})
    ax = sns.stripplot(x=args.xlabel,
                       y=args.ylabel,
                       data=data,
                       jitter=args.jitter)
    plt.xticks(rotation=args.rotate_xlabels)

    if args.ylog:
        ax.set_yscale('log', nonposy='clip')
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.out_path:
        logging.info('Saving to %s' % args.out_path)
        plt.savefig(args.out_path)
    if args.display:
        plt.show()


def plotViolinParser(subparsers):
    parser = subparsers.add_parser(
        'plotViolin',
        description='Get a "violin" plot of two fields in the db.')
    parser.set_defaults(func=plotViolin)
    parser.add_argument(
        '--sql',
        help='SQL query for TWO fields, the "x" and the "y" in the plot. '
        'Example: \'SELECT p1.value, p2.value FROM properties p1 '
        'INNER JOIN properties p2 ON p1.objectid=p2.objectid '
        'WHERE p1.key="pitch" AND p2.key="yaw"\'')
    parser.add_argument('--xlabel', required=True)
    parser.add_argument('--ylabel', required=True)
    parser.add_argument('--rotate_xlabels',
                        type=float,
                        default=0.,
                        help='Rotate labels of x-axis ticks.')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotViolin(c, args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.' % len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s' % (str(xlist), str(ylist)))

    data = pd.DataFrame({args.xlabel: xlist, args.ylabel: ylist})
    g = sns.catplot(x=args.xlabel,
                    y=args.ylabel,
                    kind="violin",
                    inner="quartiles",
                    split=True,
                    data=data)
    plt.xticks(rotation=args.rotate_xlabels)

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.out_path:
        logging.info('Saving to %s' % args.out_path)
        plt.savefig(args.out_path)
    if args.display:
        plt.show()


def plotScatterParser(subparsers):
    parser = subparsers.add_parser(
        'plotScatter', description='Get a 2d scatter plot of TWO fields.')
    parser.set_defaults(func=plotScatter)
    parser.add_argument(
        '--sql',
        help='SQL query for TWO fields, the "x" and the "y" in the plot. '
        'Example: \'SELECT p1.value, p2.value FROM properties p1 '
        'INNER JOIN properties p2 ON p1.objectid=p2.objectid '
        'WHERE p1.key="pitch" AND p2.key="yaw"\'')
    parser.add_argument('--xlabel', required=True)
    parser.add_argument('--ylabel', required=True)
    parser.add_argument('--rotate_xlabels',
                        type=float,
                        default=0.,
                        help='Rotate labels of x-axis ticks.')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotScatter(c, args):
    import matplotlib.pyplot as plt

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entry, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.' % len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s' % (str(xlist), str(ylist)))

    plt.gcf().set_size_inches(4.5, 2.4)
    plt.scatter(xlist, ylist, s=10, alpha=0.5)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.tight_layout()
    plt.xticks(rotation=args.rotate_xlabels)
    #fig.subplots_adjust(bottom=0.25)

    if args.display:
        plt.show()
    if args.out_path:
        logging.info('Saving to %s' % args.out_path)
        plt.savefig(args.out_path)


def _getImagesStats(image_entries):
    ''' Process image entries into stats. '''
    info = {}
    info['num images'] = len(image_entries)
    info['num masks'] = len(
        [x for x in image_entries if imageField(x, 'maskfile') is not None])
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
    parser.add_argument('--images_by_dir',
                        action='store_true',
                        help='Print image statistics by directory.')
    parser.add_argument('--objects_by_image',
                        action='store_true',
                        help='Print object statistics by directory.')


def printInfo(c, args):
    info = {}

    # Images stats.
    c.execute('SELECT * FROM images')
    image_entries = c.fetchall()
    if args.images_by_dir:
        # Split by image directories.
        for key, group in groupby(
                image_entries,
                lambda x: op.dirname(imageField(x, 'imagefile'))):
            info['imagedir="%s"' % key] = _getImagesStats(list(group))
    else:
        info.update(_getImagesStats(image_entries))

    # Objects stats.
    c.execute('SELECT * FROM objects')
    object_entries = c.fetchall()
    if args.objects_by_image:
        # Split by image directories.
        for key, group in groupby(object_entries,
                                  lambda x: imageField(x, 'imagefile')):
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

    pprint(info)


def dumpDbParser(subparsers):
    parser = subparsers.add_parser('dumpDb',
                                   description='Print tables of the database.')
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=['images', 'objects', 'properties', 'polygons', 'matches'],
        default=['images', 'objects', 'properties', 'polygons', 'matches'],
        help='Tables to print out, all by default.')
    parser.add_argument('--limit', type=int, help='How many entries.')
    parser.set_defaults(func=dumpDb)


def dumpDb(c, args):
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
