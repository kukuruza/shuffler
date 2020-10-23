#! /usr/bin/env python3
import argparse
import re
import datetime
import pprint
import logging


def get_parser():
    parser = argparse.ArgumentParser(
        description='''
    Read subtitles from file. Each subtitle is interpreted as an action.
    The total time per individual actions is displayed and/or written to file.
    Example: actions can be: "inspect", "label", "navigation".
    Then all subtitles in the file can be one of these three names.
    ''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--in_srt_path',
                        required=True,
                        help='The input file with subtitles in .srt format.')
    parser.add_argument(
        '--out_csv_path',
        help='If specified, write the result into this file (space delimited)')
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices=[10, 20, 30, 40],
        help='Log debug (10), info (20), warning (30), error (40).')
    return parser


def ReadAndSummarizeSrt(args):

    with open(args.in_srt_path) as f:
        lines = f.read().splitlines()

    pattern = re.compile(r'([\w\d:,]+) --> ([\w\d:,]+)')

    action_to_sec_map = {}

    for iline in range(len(lines)):

        # Find the line with times.
        line = lines[iline]
        result = re.match(pattern, line)
        if result is None:
            continue
        assert len(result.groups()) == 2, (line, len(result.groups()))
        time_from = datetime.datetime.strptime(result.groups()[0],
                                               '%H:%M:%S,%f')
        time_to = datetime.datetime.strptime(result.groups()[1], '%H:%M:%S,%f')
        sec = (time_to - time_from).total_seconds()

        # Find the action.
        iline += 1
        if iline >= len(lines):
            raise ValueError(
                "Did not find a line with subtitle text after the line with "
                "the subtitle times in the end of the file.")
        action = lines[iline]

        logging.debug('"%s" for %.3f sec', action, sec)

        # Record the action and the time it took.
        if action not in action_to_sec_map:
            action_to_sec_map[action] = 0.
        action_to_sec_map[action] += sec

    pprint.pprint(action_to_sec_map)

    # If out_txt_path is given, write to file.
    if args.out_csv_path is not None:
        with open(args.out_txt_path, 'w') as f:
            for action in action_to_sec_map:
                f.write('%s %.1f\n' % (action, action_to_sec_map[action]))


if __name__ == '__main__':
    args = get_parser().parse_args()
    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    ReadAndSummarizeSrt(args)
