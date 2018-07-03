# shuffler
Toolbox for manipulating image annotations in computer vision

# Other useful commands.
```
picsdir=4096x2160_5Hz__2018_6_15_10_10_38
ffmpeg -f image2 -r 5  -pattern_type glob -i ${picsdir}'/*.jpg' -r 5 -c:v mpeg4 -vtag xvid ${picsdir}.avi
```

# Import labelled images and display them with mask and frame_id.
```
./pipeline.py \
  importPictures --image_pattern "path/to/imagedir/*.jpg" --mask_pattern "path/to/labeldir/*.png" \
  display --labelmap_file /path/to/labelmap.json --winwidth 1500
```

# Import labelled images and write them with mask and frame_id.
```
GT_DIR=/home/evgeny/datasets/scotty/scotty_2018_6_7
./pipeline.py \
  importPictures \
    --image_pattern ${GT_DIR}"/images/*jpg" \
    --mask_pattern ${GT_DIR}"/labels/*.png" \
  writeVideo \
    --out_videopath ${GT_DIR}/visualization.avi \
    --labelmap_path ${GT_DIR}/labelmap.json \
    --fps 1 \
    --with_frameid
```

# Import labelled images and their masks and save them to a database.
```
./pipeline.py \
  -o /home/evgeny/datasets/scotty/scotty_2018_6_7/gt.db \
  importPictures \
    --image_pattern=/home/evgeny/datasets/scotty/scotty_2018_6_7/images/*jpg \
    --mask_pattern=/home/evgeny/datasets/scotty/scotty_2018_6_7/labels/*.png
```

# Evaluate MCD_DA predictions.
```
epoch=1
GT_DIR=/home/evgeny/datasets/scotty/scotty_2018_6_7
PRED_DIR=/home/evgeny/src/MCD_DA/segmentation/test_output/gta-images2scotty-train_3ch---scotty-test/MCD-normal-drn_d_105-${epoch}.tar
PRED_LABELMAP_PATH=/home/evgeny/datasets/GTA5/labelmap_GTA5.json
./pipeline.py \
  importPictures --image_pattern=${GT_DIR}/images/*.jpg --mask_pattern=${PRED_DIR}/label/*.png \
  filterWithAnother --ref_db_file=${GT_DIR}/gt.db \
  evaluateSegmentation --gt_db_file=${GT_DIR}/gt.db --gt_labelmap_path=${GT_DIR}/labelmap.json \
    --pred_labelmap_path=${PRED_LABELMAP_PATH} \
  writeVideo --out_videopath=${PRED_DIR}/vis_on_gt.avi --labelmap_path=${PRED_LABELMAP_PATH}
```