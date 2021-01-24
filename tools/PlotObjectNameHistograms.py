#! /usr/bin/env python3
import os.path as op
import pandas as pd
from argparse import ArgumentParser
import logging
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('legend', fontsize=30, handlelength=2)
matplotlib.rc('ytick', labelsize=30)


def get_parser():
    parser = ArgumentParser(
        'Plot the distribution of stamp names by database.')
    parser.add_argument('--db_paths', required=True, nargs='+')
    parser.add_argument('--legend_entries', nargs='+')
    parser.add_argument(
        '--where_objects',
        default=
        'name NOT LIKE "%page%" AND name NOT IN ("??", "??+", "+??", "reverse")'
    )
    parser.add_argument('-o', '--out_plot_path')
    parser.add_argument('--fig_width', type=int, default=60)
    parser.add_argument('--fig_height', type=int, default=10)
    parser.add_argument('--no_xticks', action='store_true')
    parser.add_argument('--fontsize', type=int, default=25)
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
        legend_entries = [op.basename(db_path) for db_path in args.db_paths]
    elif len(args.legend_entries) != len(args.db_paths):
        raise ValueError(
            'Number of elements in legend_entries (%d) and db_paths '
            '(%d) mismatches.' %
            (len(args.legend_entries), len(args.db_paths)))
    else:
        legend_entries = args.legend_entries

    # Load all the data.
    dfs = []
    for db_path, legend_entry in zip(args.db_paths, legend_entries):
        if not op.exists(db_path):
            raise FileNotFoundError('Input database "%s" not found' % db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            'SELECT name,COUNT(1) FROM objects WHERE (%s) GROUP BY name' %
            (args.where_objects, ))
        entries = c.fetchall()
        if len(entries) == 0:
            raise ValueError('No entries in database "%s"' % db_path)
        df = pd.DataFrame(entries, columns=['name', 'count'])
        df['series'] = legend_entry
        dfs.append(df)
    df = pd.concat(dfs)
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
    matplotlib.rc('legend', fontsize=args.fontsize, handlelength=2)
    matplotlib.rc('ytick', labelsize=args.fontsize)
    figsize = (args.fig_width, args.fig_height)
    df.loc[:, legend_entries].plot.bar(stacked=True, figsize=figsize)
    if args.no_xticks:
        plt.xticks([])
    else:
        plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.xlabel('')
    plt.tight_layout()
    if args.out_plot_path:
        plt.savefig(args.out_plot_path)
    if args.show:
        plt.show()


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    plot_object_name_histograms(args)


if __name__ == '__main__':
    main()
