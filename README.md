# shuffler
A toolbox for manipulating image annotations in computer vision

## Installation (using conda)

```bash
conda install imageio matplotlib lxml simplejson progressbar2 Pillow scipy
conda install -c menpo opencv
```

#### For plotting
```bash
conda install pandas seaborn
```

#### With support for testing
```bash
conda install nose scikit-image
```



## Commands

### Info

#### Print general info. Info about objects are grouped by image.
```bash
./shuffler.py \
  --in_db_file test/databases/micro1_v4.db \
  printInfo \
  --objects_by_image
```

#### Plot a histogram of value of field "yaw" of objects named "car".
```bash
./shuffler.py \
  --in_db_file test/databases/micro1_v4.db \
  plotObjectsHistogram \
  --sql_query 'SELECT value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               WHERE name="car" AND key="yaw"' \
  --xlabel yaw --display
```

#### Plot a "strip" plot of values of field "yaw" and "pitch" of objects named "car".
```bash
./shuffler.py \
  --in_db_file test/databases/micro1_v4.db \
  plotObjectsStrip \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               INNER JOIN properties p2 ON p1.objectid=p2.objectid 
               WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display
```

#### Plot a scatter plot of values of field "yaw" and "pitch" of objects named "car".
```bash
./shuffler.py \
  --in_db_file test/databases/micro1_v4.db \
  plotObjectsScatter \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               INNER JOIN properties p2 ON p1.objectid=p2.objectid 
               WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display
```

### GUI

#### Examine images.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  examineImages
```

#### Examine objects.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  examineObjects
```

#### Examine matches.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  examineMatches
```


### Filtering

#### Filter images that are present another .
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterImagesOfAnotherDb --ref_db_file test/cars/micro1_v4_singleim.db
```

#### Filter objects at image border.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterObjectsAtBorder --with_display
```

#### Filter objects that intersect other objects too much.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
filterObjectsByIntersection --with_display
```

#### Filter objects that have certain names.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterObjectsByName --good_names 'car' 'bus'
```

#### Filter objects that have low score.
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterObjectsByScore --score_threshold 0.7
```

#### Filter objects with an SQL query
```bash
./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterObjectsSQL --where 'properties.value="blue" AND objects.score > 0.8'

./shuffler.py --rootdir test \
  --in_db_file test/cars/micro1_v4.db \
  filterObjectsSQL \
  --sql 'SELECT objects.objectid FROM objects \
    INNER JOIN properties p1 ON objects.objectid=p1.objectid 
    INNER JOIN properties p2 ON objects.objectid=p2.objectid 
    WHERE p1.value="blue" AND p2.key="pitch"'
```


## Other useful commands.
```
picsdir=4096x2160_5Hz__2018_6_15_10_10_38
ffmpeg -f image2 -r 5  -pattern_type glob -i ${picsdir}'/*.jpg' -r 5 -c:v mpeg4 -vtag xvid ${picsdir}.avi
```

## Import labelled images and display them with mask and frame_id.
```
./pipeline.py \
  importPictures --image_pattern "path/to/imagedir/*.jpg" --mask_pattern "path/to/labeldir/*.png" \
  display --labelmap_file /path/to/labelmap.json --winwidth 1500
```

## Import labelled images and write them with mask and frame_id.
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

## Import labelled images and their masks and save them to a database.
```
./pipeline.py \
  -o /home/evgeny/datasets/scotty/scotty_2018_6_7/gt.db \
  importPictures \
    --image_pattern=/home/evgeny/datasets/scotty/scotty_2018_6_7/images/*jpg \
    --mask_pattern=/home/evgeny/datasets/scotty/scotty_2018_6_7/labels/*.png
```

## Evaluate MCD_DA IoU predictions.
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

## Evaluate MCD_DA and non-da ROC curve prediction
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
