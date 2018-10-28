# shuffler
A toolbox for manipulating image annotations in computer vision. Example use cases below.


## Example use cases


#### A user wants to combine [KITTI](http://www.cvlibs.net/datasets/kitti), [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd), and [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC) datasets into one ([link to code](#combine-datasets).)

A user works on object detection tasks for the autonomous car scenario, and would like to use as many annotated images as possible. In particular, they aim to combine certain classes from the datasets KITTI, BDD, and PASCAL VOC 2012. Then the combined dataset should be exported to a TF-friendly format.

#### Import from [LabelMe](http://labelme.csail.mit.edu/Release3.0), each image is labelled by multiple annotators ([link to code](#import-from-labelme).)

A user has collected a dataset of images with objects. Images were handed out to annotators who use LabelMeAnootationTool. Each image was annotated with polygons by multiple annotators for the purposes of cross-validation. A user would like to to 1) import labels from all annotators, 2) merge polygons corresponding to the same object, 3) make black-gray-white image masks, where the gray area marks the inconsistency among annotators.

#### Evaluate semantic segmentation

A neural network was trained to perform a semantic segmentation of images. We have a directory with ground truth masks and a directory with predicted masks. We would like to 1) evaluate the results, 2) write a video with images and their predicted segmentation masks side-by-side.

#### Prepare a subset of objects for training for object detection

We have a dataset with objects given as bounding boxes. We would like to remove objects on image boundary, expand bounding boxes by 10% for better training, remove objects of all types except "car", "bus", and "truck", and to remove objects smaller than 30 pixels wide. We would lile to use that subset for training.

#### Evaluate object detection on a subset of objects

In the previous use case we removed some objects for our object detection training task. Now we want to evaluate the trained model. We expect our model to detect only big objects, but we don't want to count it as a false positive if it detects a tiny object either.

#### Writing images with image crops of individual object

We have images with objects. Images have masks with those objects. We would like to crop out objects with name "car" bigger than 32 pixels wide, stretch the crops to 64x64 pixels and write a new dataset of images (and the correspodning masks)



## Installation (using conda)

Shuffler is written in Python3. The installation instructions assume Conda package management system.

```bash
conda install imageio matplotlib lxml simplejson progressbar2 Pillow scipy
conda install opencv=3.4.2  # Require opencv3.

# Add support for datasets stored as video.
conda install -c conda-forge ffmpeg=4.0

# Add support for plotting commands
conda install pandas seaborn

# Add support for unit tests
conda install nose scikit-image
```


## Commands

#### Info

```bash
# Print general info. Info about objects are grouped by image.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  printInfo --objects_by_image

# Print several tables of a database
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  dumpDb --tables objects properties

# Plot a histogram of value of field "yaw" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  plotObjectsHistogram \
  --sql_query 'SELECT value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               WHERE name="car" AND key="yaw"' \
  --xlabel yaw --display

# Plot a "strip" plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  plotObjectsStrip \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               INNER JOIN properties p2 ON p1.objectid=p2.objectid 
               WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display

# Plot a scatter plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  plotObjectsScatter \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               INNER JOIN properties p2 ON p1.objectid=p2.objectid 
               WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display
```

#### GUI

```bash
# Examine images.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  examineImages

# Examine objects.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  examineObjects

# Examine matches.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  examineMatches
```

#### Filtering

```bash
# Filter images that are present another .
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterImagesOfAnotherDb --ref_db_file 'test/cars/micro1_v4_singleim.db'

# Filter objects at image border.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsAtBorder --with_display

# Filter objects that intersect other objects too much.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
filterObjectsByIntersection --with_display

# Filter objects that have certain names.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsByName --good_names 'car' 'bus'

# Filter objects that have low score.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsByScore --score_threshold 0.7

# Filter objects with an SQL query
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsSQL --where 'properties.value="blue" AND objects.score > 0.8'
# or
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsSQL \
  --sql 'SELECT objects.objectid FROM objects \
    INNER JOIN properties p1 ON objects.objectid=p1.objectid 
    INNER JOIN properties p2 ON objects.objectid=p2.objectid 
    WHERE p1.value="blue" AND p2.key="pitch"'
```

#### Modifications

```bash
# Add a video of images and a video of masks.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  addVideo --image_video_path 'test/moon/images.avi' --mask_video_path 'test/moon/masks.avi'

# Add image and mask pictures.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  addPictures --image_pattern 'test/moon/images/*.jpg' --mask_pattern 'test/moon/masks/*.png'

# Discard all, but the first N images.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  headImages -n 2

# Discard all, but the last N images.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  tailImages -n 2

# Expand bounding boxes
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  expandBoxes --expand_perc 0.2 --with_display
# or to try match the target_ratio
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  expandBoxes --expand_perc 0.2 --target_ratio 0.75 --with_display

# To move the directory of images or masks
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  moveDir --image_dir '/tmp/images' --mask_dir '/tmp/masks' --where_image 'imagefile LIKE "cars/images/%"'

# Add a new database (follow this one with mergeObjectDuplicates).
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  addDb --db_file 'test/cars/micro1_v4_singleim.db'

# Merge multiple objects that have the same ROI 
# For example, when objects were collected from different annotators and then merged via addDb.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  mergeObjectDuplicates

# Split into several databases.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  splitDb --out_dir '.' --out_names 'test' 'train' --out_fractions 0.3 0.7 --shuffle

# Masks from polygons.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  polygonsToMask --mask_pictures_dir 'cars/mask_polygons' --skip_empty_masks
```

#### Evaluation of ML tasks
```bash
# Evaluate semantic segmentation.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  evaluateSegmentationIoU --gt_db_file 'test/cars/micro1_v4_polygons.db' \
    --gt_mapping_dict '{0: "background", 255: "car"}' --out_dir 'test/testIoU'

# Evaluate single-class detection (background vs foreground), 
# where masks of current db are grayscale probabilities.
./shuffler.py --rootdir 'test' --in_db_file 'test/cars/micro1_v4.db' \
  evaluateBinarySegmentation --gt_db_file 'test/cars/micro1_v4_polygons.db'

# Evaluate object detection.
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  evaluateDetection --gt_db_file 'test/cars/micro1_v4_singleim.db'
```

#### Write a new image directory / video

```bash
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  writeImages --out_pictures_dir 'cars/exported' --mask_alpha 0.5 --with_imageid

./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  cropObjects --image_pictures_dir 'cars/exported' --mask_pictures_dir 'cars/exportedmask' \
  --target_width 64 --target_height 64
```

### Imports

#### Import KITTI
```bash
./shuffler.py --rootdir ${KITTI} \
  -o '/tmp/kitti.db' \
  importKitti \
  --images_dir=${KITTI}/data_semantics/training/image_2  \
  --segmentation_dir=${KITTI}/data_semantics/training/instance

./shuffler.py --rootdir ${KITTI} \
  -o '/tmp/kitti.db' \
  importKitti \
  --images_dir=${KITTI}/data_semantics/training/image_2  \
  --detection_dir=${KITTI}/data_object_image_2/training/label_2
```

#### Import LabelMe
```bash
./shuffler.py --rootdir '.' \
  -i 'test/labelme/init.db' \
  importLabelme --annotations_dir 'test/labelme/w55-e04-images1'

./shuffler.py --rootdir '.' \
  -i test/labelme/init.db \
  importLabelmeObjects --annotations_dir test/labelme/w55-e04-objects1 \
  --keep_original_object_name --polygon_name objects1
```

### Import Pascal
```bash
./shuffler.py --rootdir ${VOC2012} \
  -o '/tmp/pascal.db' \
  importPascalVoc2012 \
  --images_dir ${VOC2012}/JPEGImages \
  --detection_dir ${VOC2012}/Annotations \
  --segmentation_dir ${VOC2012}/SegmentationClass
```

### Chaining commands

Sub-commands can be chained via the special symbol "\|" (here, the backslash escapes the following vertical bar from a Unix shell.)

```bash
./shuffler.py --rootdir 'test' --in_db_file 'test/cars/micro1_v4.db' \
  addVideo --image_video_path 'test/moon/images.avi' --mask_video_path 'test/moon/masks.avi' \| \
  printInfo \| \
  moveDir --image_dir 'test/cars/images' --where_image 'imagefile LIKE "cars/images/%"' \| \
  dumpDb --tables 'images' 'objects'

./shuffler.py --rootdir 'test' --in_db_file 'test/cars/micro1_v4.db' \
  addDb --db_file 'test/cars/micro1_v4_singleim.db' \| \
  mergeObjectDuplicates \| \
  polygonsToMask --mask_pictures_dir 'cars/mask_polygons' --skip_empty_masks \| \
  dumpDb --tables images \| \
  examineImages --mask_alpha 0.5
```

## Code for the use cases

#### <a name="combine-datasets">Combine KITTI, BDD, and PASCAL VOC 2012 datasets into one</a>

```bash
KITTI=/path/to/directory/KITTI
VOC2012=/path/to/directory/VOC2012

./shuffler.py --rootdir ${KITTI} \
  -o '/tmp/kitti.db' \
  importKitti \
  --images_dir=${KITTI}/data_semantics/training/image_2  \
  --detection_dir=${KITTI}/data_object_image_2/training/label_2
```

#### <a name="import-from-labelme">Import from LabelMe, each image is labelled by multiple annotators.</a>

```bash
./shuffler.py --rootdir '.' -i 'test/labelme/init.db' \
  importLabelmeObjects --annotations_dir 'test/labelme/w55-e04-objects1' \
  --keep_original_object_name --polygon_name annotator1 \| \
  importLabelmeObjects --annotations_dir 'test/labelme/w55-e04-objects2' \
  --keep_original_object_name --polygon_name annotator2 \| \
  importLabelmeObjects --annotations_dir 'test/labelme/w55-e04-objects3' \
  --keep_original_object_name --polygon_name annotator3 \| \
  importLabelmeObjects --annotations_dir 'test/labelme/w55-e04-objects4' \
  --keep_original_object_name --polygon_name annotator4 \| \
  mergeObjectDuplicates \| \
  polygonsToMask --mask_pictures_dir 'test/labelme/mask_polygons' --skip_empty_masks \| \
  dumpDb --tables objects polygons \| \
  examineImages --mask_aside
```




## Testing code

### Run all unit tests
```bash
python3 -m "nose"
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
