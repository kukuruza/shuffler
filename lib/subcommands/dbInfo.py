import os.path as op
import logging
import pprint
from itertools import groupby

from lib.backend import backendDb


def add_parsers(subparsers):
    plotHistogramParser(subparsers)
    plotScatterParser(subparsers)
    plotStripParser(subparsers)
    plotViolinParser(subparsers)
    printInfoParser(subparsers)
    dumpDbParser(subparsers)
    diffDbParser(subparsers)


def _updateFont(fontsize):
    import matplotlib
    matplotlib.rc('legend', fontsize=fontsize, handlelength=2)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)


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
    sql_group = parser.add_mutually_exclusive_group()
    sql_group.add_argument(
        '--sql',
        help='SQL query for ONE field, the "x" in the plot. '
        'Example: \'SELECT value FROM properties WHERE key="pitch"\'')
    sql_group.add_argument(
        '--sql_stacked',
        help='SQL query for TWO fields, the "x" in the plot, and the "series". '
        'Will plot stacked histograms, one for each "series". Example: '
        '\'SELECT value,key FROM properties WHERE key IN ("yaw", "pitch")\'. '
        'Here, there will be two stacked histograms, "yaw" and "pitch". '
        'Ordering of histograms is not supported.')
    parser.add_argument('--xlabel')
    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--bins', type=int)
    parser.add_argument('--xlim', type=float, nargs='+')
    parser.add_argument('--colormap', default='Spectral')
    parser.add_argument('--rotate_xlabels',
                        type=float,
                        default=0.,
                        help='Rotate labels of x-axis ticks.')
    parser.add_argument('--categorical', action='store_true')
    parser.add_argument('--fontsize', type=int, default=15)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotHistogram(c, args):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd
    _updateFont(args.fontsize)

    c.execute(args.sql if args.sql else args.sql_stacked)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if args.sql and len(entries[0]) != 1:
        raise ValueError('Must query for 1 fields, not %d.' % len(entries[0]))
    elif args.sql_stacked and len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))

    # Make a dataframe.
    columns = ['x'] if args.sql else ['x', 'series']
    df = pd.DataFrame(entries, columns=columns)
    df['x'] = pd.to_numeric(df['x'], downcast='integer')
    logging.debug(df)

    if args.sql_stacked and not args.categorical:
        df.pivot_table(index='x', columns='series', aggfunc='size').\
           plot.bar(stacked=True, cmap=cm.get_cmap(args.colormap))
        # Put a legend to the right of the current axis
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.7, box.height])
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
    elif args.sql and not args.categorical:
        df['x'].plot.hist()
    elif args.sql and args.categorical:
        import seaborn as sns
        sns.countplot(x=args.xlabel, df=df, order=df['x'].value_counts().index)
    elif args.sql_stacked and args.categorical:
        raise NotImplementedError()
    else:
        assert 0, "Should not be here."

    plt.xticks(rotation=args.rotate_xlabels)
    if args.xlim:
        if not len(args.xlim) == 2:
            raise Exception('Argument xlim requires to numbers')
        plt.xlim(args.xlim)
    if args.ylog:
        plt.yscale('log', nonposy='clip')
    if args.xlabel:
        plt.xlabel(args.xlabel, fontsize=args.fontsize)
    plt.ylabel('')
    plt.tight_layout()
    if args.out_path:
        logging.info('Saving to %s', args.out_path)
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
    parser.add_argument('--fontsize', type=int, default=15)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotStrip(c, args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    _updateFont(args.fontsize)

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.', len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s', str(xlist), str(ylist))

    _, ax = plt.subplots()
    data = pd.DataFrame({args.xlabel: xlist, args.ylabel: ylist})
    ax = sns.stripplot(x=args.xlabel,
                       y=args.ylabel,
                       data=data,
                       jitter=args.jitter)
    plt.xticks(rotation=args.rotate_xlabels)

    if args.ylog:
        ax.set_yscale('log', nonposy='clip')
    plt.xlabel(args.xlabel, fontsize=args.fontsize)
    plt.ylabel(args.ylabel)
    if args.out_path:
        logging.info('Saving to %s', args.out_path)
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
    parser.add_argument('--fontsize', type=int, default=15)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotViolin(c, args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    _updateFont(args.fontsize)

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entries, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.', len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s', str(xlist), str(ylist))

    data = pd.DataFrame({args.xlabel: xlist, args.ylabel: ylist})
    sns.catplot(x=args.xlabel,
                y=args.ylabel,
                kind="violin",
                inner="quartiles",
                split=True,
                data=data)
    plt.xticks(rotation=args.rotate_xlabels)

    plt.xlabel(args.xlabel, fontsize=args.fontsize)
    plt.ylabel(args.ylabel, fontsize=args.fontsize)
    if args.out_path:
        logging.info('Saving to %s', args.out_path)
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
    parser.add_argument('--fontsize', type=int, default=15)
    parser.add_argument('--tick_base',
                        type=int,
                        help='Sets the distance between ticks. '
                        'The aspect ratio of X and Y axes is set to 1.')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--out_path')


def plotScatter(c, args):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _updateFont(args.fontsize)

    c.execute(args.sql)
    entries = c.fetchall()

    # Clean data.
    if not entries:
        logging.info('No entry, nothing to draw.')
        return
    if len(entries[0]) != 2:
        raise ValueError('Must query for 2 fields, not %d.' % len(entries[0]))
    xylist = [(x, y) for (x, y) in entries if x is not None and y is not None]
    logging.info('%d entries have both fields non-None.', len(xylist))

    # From a list of tuples to two lists.
    xlist, ylist = tuple(map(list, zip(*xylist)))
    xlist = _maybeNumerizeProperty(xlist)
    ylist = _maybeNumerizeProperty(ylist)
    logging.debug('%s\n%s', str(xlist), str(ylist))

    #    plt.gcf().set_size_inches(4.5, 2.4)
    plt.scatter(xlist, ylist, s=10, alpha=0.5)
    plt.xlabel(args.xlabel, fontsize=args.fontsize)
    plt.ylabel(args.ylabel, fontsize=args.fontsize)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.xticks(rotation=args.rotate_xlabels)
    if args.tick_base:
        loc = ticker.MultipleLocator(base=args.tick_base)
        plt.gca().xaxis.set_major_locator(loc)
        plt.gca().yaxis.set_major_locator(loc)

    if args.display:
        plt.show()
    if args.out_path:
        logging.info('Saving to %s', args.out_path)
        plt.savefig(args.out_path)


def _getImagesStats(image_entries):
    ''' Process image entries into stats. '''
    info = {}
    info['num images'] = len(image_entries)
    info['num masks'] = len([
        x for x in image_entries
        if backendDb.imageField(x, 'maskfile') is not None
    ])
    # Width.
    widths = set([backendDb.imageField(x, 'width') for x in image_entries])
    if len(widths) > 1:
        info['image width'] = '%s different values' % len(widths)
    elif len(widths) == 1:
        info['image width'] = '%s' % next(iter(widths))
    # Height.
    heights = set([backendDb.imageField(x, 'height') for x in image_entries])
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
                lambda x: op.dirname(backendDb.imageField(x, 'imagefile'))):
            info['imagedir="%s"' % key] = _getImagesStats(list(group))
    else:
        info.update(_getImagesStats(image_entries))

    # Objects stats.
    c.execute('SELECT * FROM objects')
    object_entries = c.fetchall()
    if args.objects_by_image:
        # Split by image directories.
        for key, group in groupby(
                object_entries,
                lambda x: backendDb.imageField(x, 'imagefile')):
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

    pprint.pprint(info)


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


def diffDbParser(subparsers):
    parser = subparsers.add_parser(
        'diffDb',
        description='Compute the diff between the open and the reference db. '
        'Print the results, and return them as a dict.')
    parser.set_defaults(func=diffDb)
    parser.add_argument('--ref_db_file',
                        required=True,
                        help='The database to take diff against.')
    parser.add_argument('--where_object',
                        default='TRUE',
                        help='SQL "where" clause for the "objects" table.')


def diffDb(c, args):
    results = {}

    c.execute('ATTACH ? AS "ref"', (args.ref_db_file, ))

    # Get imagefiles.
    c.execute('SELECT imagefile FROM images')
    imagefiles = c.fetchall()
    c.execute('SELECT imagefile FROM ref.images')
    imagefiles_ref = c.fetchall()

    # Imagefile statistics.
    imagefiles_both = set(imagefiles) & set(imagefiles_ref)
    imagefiles_new = set(imagefiles) - set(imagefiles_ref)
    imagefiles_old = set(imagefiles_ref) - set(imagefiles)
    results['images'] = {
        'remaining': len(imagefiles_both),
        'new': len(imagefiles_new),
        'old': len(imagefiles_old),
    }

    # Object statistics.
    c.execute(
        'SELECT COUNT(1) FROM objects obj INNER JOIN ref.objects ref_obj '
        'ON obj.objectid=ref_obj.objectid AND (%s)' % args.where_object)
    num_remaining = c.fetchone()[0]
    c.execute('SELECT COUNT(1) FROM objects obj WHERE objectid NOT IN '
              '(SELECT objectid FROM ref.objects) AND (%s)' %
              args.where_object)
    num_new = c.fetchone()[0]
    c.execute('SELECT COUNT(1) FROM ref.objects obj WHERE objectid NOT IN '
              '(SELECT objectid FROM objects) AND (%s)' % args.where_object)
    num_old = c.fetchone()[0]
    c.execute(
        'SELECT COUNT(1) FROM objects obj INNER JOIN ref.objects ref_obj '
        'ON obj.objectid=ref_obj.objectid '
        'WHERE (obj.x1 != ref_obj.x1 OR obj.y1 != ref_obj.y1 OR '
        'obj.width != ref_obj.width OR obj.height != ref_obj.height) AND (%s)'
        % args.where_object)
    num_moved = c.fetchone()[0]
    c.execute(
        'SELECT COUNT(1) FROM objects obj INNER JOIN ref.objects ref_obj '
        'ON obj.objectid=ref_obj.objectid WHERE (obj.name == ref_obj.name) AND (%s)'
        % args.where_object)
    num_not_renamed = c.fetchone()[0]
    results['objects'] = {
        'remaining': num_remaining,
        'new': num_new,
        'old': num_old,
        'remaining_moved': num_moved,
        'remaining_not_renamed': num_not_renamed,
    }

    pprint.pprint(results)

    # FIXME: This always throws an exception that the database is locked.
    #c.execute('DETACH DATABASE "ref"')
    return results
