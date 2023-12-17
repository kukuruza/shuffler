#! /usr/bin/env python3
import sys
import os.path as op
import pandas as pd
import argparse
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from shuffler.operations import evaluate
from shuffler.backend import backend_db


def get_parser():
    parser = argparse.ArgumentParser(
        'Plot the recall of each class for several classification results.')
    parser.add_argument('--evaluated_db_paths', required=True, nargs='+')
    parser.add_argument('--gt_db_path', required=True)
    parser.add_argument('--legend_entries', nargs='+')
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


def compare_multiple_classifications(args):
    # Process empty and non-empty legend_entries.
    if args.legend_entries is None:
        series = [
            'campaign %s' % op.basename(x) for x in args.evaluated_db_paths
        ]
    elif len(args.legend_entries) != len(args.evaluated_db_paths):
        raise ValueError(
            'Number of elements in legend_entries (%d) and evaluated_db_paths '
            '(%d) mismatches.' %
            (len(args.legend_entries), len(args.evaluated_db_paths)))
    else:
        series = args.legend_entries

    dfs = []
    for evaluated_db_path, legend_entry in zip(args.evaluated_db_paths,
                                               series):
        if not op.exists(evaluated_db_path):
            raise FileNotFoundError('Database "%s" not found.' %
                                    evaluated_db_path)
        conn = backend_db.connect(evaluated_db_path, 'load_to_memory')
        c = conn.cursor()

        args1 = argparse.Namespace(rootdir=None,
                                   gt_db_file=args.gt_db_path,
                                   out_directory=None)
        result = evaluate.evaluateClassification(c, args1)
        conn.close()

        result = [(k, v['support'], v['recall']) for k, v in result.items()
                  if k not in ['accuracy', 'macro avg', 'weighted avg']]
        df = pd.DataFrame(result, columns=['name', 'count', 'recall'])
        df['series'] = legend_entry

        dfs.append(df)
    df = pd.concat(dfs)

    # Scale the recall of all series by the average count of data to illustrate
    # class count visually.
    df = df.join(df.groupby(["name", "series", "recall"])["count"].mean(),
                 on=["name", "series", "recall"],
                 rsuffix='_mean').drop('count', axis=1)
    df['recall'] *= df['count_mean']
    print(df)

    # Maybe keep only those classes with enough objects.
    if args.at_least:
        df = df[df['count_mean'] > args.at_least]

    grouped = df.groupby(["name", "count_mean", "series"])[["recall"]].mean()
    df = grouped.unstack().sort_values('count_mean',
                                       ascending=False).reset_index()
    print(df)

    # TODO: keep only one dataframe.
    df_count = df.copy()
    df.columns = df.columns.get_level_values(1)
    df_count.columns = df_count.columns.get_level_values(0)
    print('df.columns: ', df.columns)
    print('df_count.columns: ', df_count.columns)

    # Plot.
    figsize = (args.fig_width, args.fig_height)
    matplotlib.rc('legend', fontsize=args.fontsize + 2, handlelength=2)
    matplotlib.rc('legend', title_fontsize=args.fontsize + 4)
    matplotlib.rc('ytick', labelsize=args.fontsize + 2)
    df.plot.bar(ax=ax, y=series, figsize=figsize, alpha=0.6)
    ax = df_count.plot.bar(x='name', y='count_mean', color='gray', alpha=0.2)
    if args.ylog:
        plt.yscale('log', nonpositive='clip')
    if args.no_xticks:
        plt.xticks([])
    else:
        plt.xticks(rotation=90)
    plt.grid(axis='y')
    # plt.xlabel('')
    if args.title is not None:
        plt.title(args.title, fontsize=args.fontsize + 6, pad=30)
    plt.tight_layout()
    plt.gca().get_legend().set_title('Model')
    if args.out_plot_path:
        plt.savefig(args.out_plot_path)
    if args.show:
        plt.show()


def main(argv):
    args = get_parser().parse_args(argv)
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    compare_multiple_classifications(args)


if __name__ == '__main__':
    main(sys.argv[1:])
