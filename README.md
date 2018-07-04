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

./pipeline.py \
  importPictures \
    --image_pattern ${GT_DIR}"/images/*jpg" \
    --mask_pattern ${GT_DIR}"/labels/*.png" \
  writePictures \
    --out_dir ${GT_DIR}/visualization \
    --labelmap_path ${GT_DIR}/labelmap.json \
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

# Evaluate MCD_DA IoU predictions.
```
epoch=4
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

# Evaluate MCD_DA and non-da ROC curve prediction
```
epoch=4
# adapt
PRED_DIR=/home/evgeny/src/MCD_DA/segmentation/test_output/gta-images2scotty-train_3ch---scotty-test42/MCD-normal-drn_d_105-${epoch}.tar/prob
# no adapt
PRED_DIR=/home/evgeny/src/MCD_DA/segmentation/test_output/gta-images_only_3ch---scotty-test42/normal-drn_d_105-${epoch}.tar/prob
# alex weiss net
PRED_DIR=/home/evgeny/datasets/scotty/scotty_2018_6_7_pred_alex/segout

./pipeline.py \
  importPictures --image_pattern=${GT_DIR}"/images/*.jpg" --mask_pattern=${PRED_DIR}/"*.png" 
  evaluateSegmentationROC --gt_db_file=${GT_DIR}/gt.db --out_dir ${PRED_DIR}
```

# Make grid out of four videos.
```
python tools/multiplepictures2video.py -i \
  "/home/evgeny/datasets/scotty/scotty_2018_6_7/visualization/*.jpg" \
  "/home/evgeny/datasets/scotty/scotty_2018_6_7_pred_alex/segout/*.png" \
  "/home/evgeny/src/MCD_DA/segmentation/test_output/gta-images2scotty-train_3ch---scotty-test42/MCD-normal-drn_d_105-4.tar/prob/*.png" \
  "/home/evgeny/src/MCD_DA/segmentation/test_output/gta-images_only_3ch---scotty-test42/normal-drn_d_105-4.tar/prob/*.png" \
  -o /home/evgeny/Desktop/grid.avi \
  --imwidth 1500 --fps 1 --overwrite
```