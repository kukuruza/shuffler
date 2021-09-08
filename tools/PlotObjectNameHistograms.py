#! /usr/bin/env python3
import sys
import os.path as op
import pandas as pd
import argparse
import logging
import sqlite3
import matplotlib.pyplot as plt
import matplotlib


def get_parser():
    parser = argparse.ArgumentParser(
        'Plot the distribution of stamp names by database.')
    parser.add_argument('--db_path', required=True)
    parser.add_argument('--campaign_names', required=True, nargs='+')
    parser.add_argument('--legend_entries', nargs='+')
    parser.add_argument(
        '--where_objects',
        default=
        'name NOT LIKE "%page%" AND name NOT IN ("??", "??+", "+??", "reverse")'
    )
    parser.add_argument('-o', '--out_plot_path')
    parser.add_argument(
        '--at_least',
        type=int,
        help='If specififed, will only display classes with at_least instences.'
    )
    parser.add_argument('--title', 
        help='If specified, will add this text as caption.')
    parser.add_argument('--fig_width', type=int, default=60)
    parser.add_argument('--fig_height', type=int, default=10)
    parser.add_argument('--no_xticks', action='store_true')
    parser.add_argument('--fontsize', type=int, default=25)
    parser.add_argument('--ylog', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def plot_object_name_histograms(args):
    # Process empty and non-empty legend_entries.
    if args.legend_entries is None:
        legend_entries = [name for name in args.campaign_names]
    elif len(args.legend_entries) != len(args.campaign_names):
        raise ValueError(
            'Number of elements in legend_entries (%d) and campaign_names '
            '(%d) mismatches.' %
            (len(args.legend_entries), len(args.campaign_names)))
    else:
        legend_entries = args.legend_entries

    # Load all the data.
    if not op.exists(args.db_path):
        raise FileNotFoundError('Database "%s" not found.' % args.db_path)
    conn = sqlite3.connect(args.db_path)
    c = conn.cursor()

    dfs = []
    for campaign_name, legend_entry in zip(args.campaign_names,
                                           legend_entries):
        c.execute(
            'SELECT name,COUNT(1) FROM objects o '
            'JOIN properties p ON o.objectid=p.objectid '
            'WHERE key="campaign" AND value=? AND (%s) GROUP BY name' %
            args.where_objects, (campaign_name, ))
        entries = c.fetchall()
        if len(entries) == 0:
            raise ValueError('No entries for campaign "%s"' % campaign_name)

        df = pd.DataFrame(entries, columns=['name', 'count'])
        df['series'] = legend_entry
        dfs.append(df)
    df = pd.concat(dfs)

    conn.close()
    # print(df)

    # Maybe keep only those classes with enough objects.
    if args.at_least:
        df = df.groupby(['name'
                         ]).filter(lambda x: x['count'].sum() >= args.at_least)

    # Transform and plot.
    df = df.pivot_table(index='name',
                        columns='series',
                        values='count',
                        aggfunc='sum',
                        fill_value=0,
                        margins=True).sort_values(
                            'All', ascending=False).drop('All').drop('All',
                                                                     axis=1)
    print(df)

    # Plot.
    matplotlib.rc('legend', fontsize=args.fontsize-2, handlelength=2)
    matplotlib.rc('legend', title_fontsize=args.fontsize-2)
    matplotlib.rc('ytick', labelsize=args.fontsize)
    figsize = (args.fig_width, args.fig_height)
    df.loc[:, legend_entries].plot.bar(stacked=True, figsize=figsize)
    if args.ylog:
        plt.yscale('log', nonpositive='clip')
    if args.no_xticks:
        plt.xticks([])
    else:
        plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.xlabel('')
    if args.title is not None:
        plt.title(args.title, fontsize=args.fontsize + 2, pad=30)
    plt.tight_layout()
    plt.gca().get_legend().set_title('active learning cycle')
    if args.out_plot_path:
        plt.savefig(args.out_plot_path)
    if args.show:
        plt.show()


def main(argv):
    args = get_parser().parse_args(argv)
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    plot_object_name_histograms(args)


if __name__ == '__main__':
    main(sys.argv[1:])
