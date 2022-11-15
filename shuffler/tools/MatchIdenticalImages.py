#! /usr/bin/env python3
import os, os.path as op
import argparse
import glob
import progressbar

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path_pattern1", required=True,
        help="Pattern of path 'path/to/files/1/*.jpg'. Dont forget quotes.")
    parser.add_argument("--file_path_pattern2", required=True,
        help="Pattern of path 'path/to/files/1/*.jpg'. Dont forget quotes.")
    parser.add_argument("--output_file", required=True,
        help="Path of file where to write the matches.")
    return parser

def read_binfile(path):
    with open(path, 'rb') as f:
        return f.read()

def read_bytes_from_paths(file_paths):
    print("Found %d files" % len(file_paths))
    return [read_binfile(file_path) for file_path in progressbar.progressbar(file_paths)]

if __name__ == "__main__":
    args = get_parser().parse_args()

    file_paths1 = glob.glob(args.file_path_pattern1)
    file_paths2 = glob.glob(args.file_path_pattern2)

    images1 = read_bytes_from_paths(file_paths1)
    images2 = read_bytes_from_paths(file_paths2)

    if len(images1) <= 1 or len(images2) <= 1:
        raise Exception('Too few images.')

    tuples = []
    found_ids1 = set()
    found_ids2 = set()

    for i1, image1 in progressbar.progressbar(enumerate(images1)):
        for i2, image2 in enumerate(images2):
            # "i1 not in found_ids2" to speed up.
            if i1 not in found_ids2 and image1 == image2:
                tuples.append((i1, i2))
                found_ids1.add(i1)
                found_ids2.add(i2)
                continue

    for i1 in range(len(images1)):
        if i1 not in found_ids1:
            tuples.append((i1, None))

    for i2 in range(len(images2)):
        if i2 not in found_ids2:
            tuples.append((None, i2))

    with open(args.output_file, 'w') as f:
        for i1, i2 in tuples:
            name1 = '-' if i1 is None else op.basename(file_paths1[i1])
            name2 = '-' if i2 is None else op.basename(file_paths2[i2])
            s = '%s\t%s' % (name1, name2)
            print (s)
            f.write('%s\n' % s)
