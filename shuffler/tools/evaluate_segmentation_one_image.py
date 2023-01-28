#! /usr/bin/env python3

import numpy as np
import imageio
import cv2
import argparse
import logging
import pandas as pd

from shuffler.utils import general as general_utils
from shuffler.utils import parser as parser_utils
from shuffler.operations import evaluate


def getParser():
    parser = argparse.ArgumentParser(
        "Evaluate segmentation result. Takes a predicted mask and a ground truth mask."
    )
    parser.add_argument(
        '-pred',
        '--predicted_path',
        required=True,
        help='Path of prediction mask, has to be a grayscale image.')
    parser.add_argument(
        '-gt',
        '--ground_truth_path',
        required=True,
        help='Path of ground truth mask, has to be a grayscale image.')
    parser.add_argument(
        '--gt_mapping_dict',
        required=True,
        help=
        'the mapping from ground truth maskfile to classes. E.g. "{0: \'background\', 255: \'car\'}"'
    )
    parser.add_argument(
        '--pred_mapping_dict',
        help=
        'the mapping from predicted mask to classes, if different from "gt_mapping_dict"'
    )
    parser.add_argument(
        '--logging',
        default=20,
        type=int,
        choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
    parser_utils.addMaskMappingArgument(parser)
    return parser


def evaluateSegmentationOneImage(args):

    labelmap_gt, labelmap_pr, class_names = evaluate._label2classMapping(
        args.gt_mapping_dict, args.pred_mapping_dict)

    mask_pr = imageio.imread(args.predicted_path).astype(float)
    mask_gt = imageio.imread(args.ground_truth_path).astype(float)
    if len(mask_pr.shape) == 3:
        mask_pr = mask_pr[:, :, 0]
    if len(mask_gt.shape) == 3:
        mask_gt = mask_gt[:, :, 0]

    mask_gt = general_utils.applyMaskMapping(mask_gt, labelmap_gt)
    mask_pr = general_utils.applyMaskMapping(mask_pr, labelmap_pr)
    assert mask_gt.dtype == float
    assert mask_pr.dtype == float
    mask_pr = cv2.resize(mask_pr, (mask_gt.shape[1], mask_gt.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    careabout = ~np.isnan(mask_gt)

    mask_pr[~careabout] = 0.5
    mask_gt[~careabout] = 0.5
    cv2.imshow(
        'mask_gt',
        cv2.resize(mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST))
    cv2.imshow(
        'mask_pr',
        cv2.resize(mask_pr, (512, 512), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(-1)

    mask_gt = mask_gt[careabout][:].astype(int)
    mask_pr = mask_pr[careabout][:].astype(int)
    hist = evaluate.fast_hist(mask_gt, mask_pr, len(class_names))

    # Get label distribution.
    pr_per_class = hist.sum(0)
    gt_per_class = hist.sum(1)

    iou_list = evaluate.per_class_iu(hist)
    fwIoU = evaluate.calc_fw_iu(hist)
    pixAcc = evaluate.calc_pixel_accuracy(hist)
    mAcc = evaluate.calc_mean_accuracy(hist)

    result_df = pd.DataFrame({
        'class': class_names,
        'IoU': iou_list,
        "pr_distribution": pr_per_class,
        "gt_distribution": gt_per_class,
    })
    result_df["IoU"] *= 100  # Changing to percent ratio.

    result_df.set_index("class", inplace=True)
    print("---- info per class -----")
    print(result_df)

    result_ser = pd.Series({
        "pixAcc": pixAcc,
        "mAcc": mAcc,
        "fwIoU": fwIoU,
        "mIoU": iou_list.mean()
    })
    result_ser = result_ser[["pixAcc", "mAcc", "fwIoU", "mIoU"]]
    result_ser *= 100  # change to percent ratio

    print("---- result summary -----")
    print(result_ser)


if __name__ == '__main__':
    args = getParser().parse_args()

    logging.basicConfig(level=args.logging,
                        format='%(levelname)s: %(message)s')

    evaluateSegmentationOneImage(args)
