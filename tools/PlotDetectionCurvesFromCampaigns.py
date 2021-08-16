import os.path as op
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_parser():
    parser = argparse.ArgumentParser(
        'Plot the precision-recall curve with data from several campaigns.')
    parser.add_argument('--campaigns_dir', required=True)
    parser.add_argument(
        '--curve_path_pattern',
        default=
        'campaign%d/detected-trained-on-campaign3to%d/campaign%d-1800x1200.stamps/precision-recall-stamp.txt',
        help='Relative to campaign_dir.')
    parser.add_argument('--main_campaign_id', type=int, required=True)
    parser.add_argument('--campaign_ids', type=int, required=True, nargs='+')
    parser.add_argument('-o', '--out_plot_path')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def plot_detection_curves_from_campaigns(args):

    for campaign_id in args.campaign_ids:
        curve_path = op.join(
            args.campaigns_dir, args.curve_path_pattern %
            (args.main_campaign_id, campaign_id, args.main_campaign_id))
        logging.info('Parsing: %s', curve_path)
        df = pd.read_csv(curve_path, sep=' ', header=0)
        label = ('trained on cycle 1' if campaign_id == 3 else
                 'trained on cycles 1-%d' % (campaign_id - 2))
        plt.plot(df['recall'], df['precision'], label=label)

    plt.legend(loc='lower left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    ax = plt.gca()
    ax.grid(which='major', linewidth='0.5')
    ax.grid(which='minor', linewidth='0.2')
    loc = ticker.MultipleLocator(0.2)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    loc = ticker.MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if args.out_plot_path:
        plt.savefig(args.out_plot_path)
    if args.show:
        plt.show()


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    plot_detection_curves_from_campaigns(args)


if __name__ == '__main__':
    main()
