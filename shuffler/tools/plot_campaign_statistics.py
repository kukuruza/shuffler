import sqlite3
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        'Plot the precision-recall curve with data from several campaigns.')
    parser.add_argument('--all_db_file', required=True)
    parser.add_argument('--in_db_file', required=True)
    parser.add_argument('-o', '--out_plot_path')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def plot_campaign_statistics(args):

    conn = sqlite3.connect('file:%s?mode=ro' % args.all_db_file, uri=True)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(imagefile) FROM images')
    num_all_images = cursor.fetchone()[0]
    conn.close()

    conn = sqlite3.connect('file:%s?mode=ro' % args.in_db_file, uri=True)
    cursor = conn.cursor()
    cursor.execute('SELECT CAST(name AS INT),COUNT(imagefile) '
                   'FROM images '
                   'WHERE name IS NOT NULL '
                   'GROUP BY name '
                   'ORDER BY CAST(name AS INT) ASC ')
    entries = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(entries, columns=['campaign', 'count'])
    df['cycle'] = df['campaign'] - 2
    df['cycle'] = df['cycle'].astype('string')

    df = df.append(
        {
            'cycle': 'unlabeled',
            'count': num_all_images - df['count'].sum()
        },
        ignore_index=True)

    print(df)
    print(df['count'].sum())
    df['labels'] = df['cycle'] + ":  " + df['count'].astype('string')
    df.loc[df.index[-2],
           'labels'] = df.loc[df.index[-2], 'labels'] + " (in progress)"

    patches, _ = plt.pie(df['count'])
    plt.legend(patches, df['labels'], loc='lower left')

    plt.tight_layout()
    if args.out_plot_path:
        plt.savefig(args.out_plot_path)
    if args.show:
        plt.show()


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    plot_campaign_statistics(args)


if __name__ == '__main__':
    main()
