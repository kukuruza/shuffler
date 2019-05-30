## Commands

### <a name="import">Imports
 
Import](#import) most common computer vision datasets. Currently support formats of [KITTI](http://www.cvlibs.net/datasets/kitti), [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd), and [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC), and LabelMe. The list is growing, pull requests are welcome.

#### Import KITTI
```bash
# Import KITTI segmentation
./shuffler.py --rootdir ${KITTI} \
  -o '/tmp/kitti.db' \
  importKitti \
  --images_dir=${KITTI}/data_semantics/training/image_2  \
  --segmentation_dir=${KITTI}/data_semantics/training/instance

# Import KITTI detection (does not share images with segmentation)
./shuffler.py --rootdir ${KITTI} \
  -o '/tmp/kitti.db' \
  importKitti \
  --images_dir=${KITTI}/data_semantics/training/image_2  \
  --detection_dir=${KITTI}/data_object_image_2/training/label_2

# Import LabelMe.
./shuffler.py --rootdir '.' \
  -i 'test/labelme/init.db' \
  importLabelme --annotations_dir 'test/labelme/w55-e04-images1'

# Import LabelMe, objects by ids.
./shuffler.py --rootdir '.' \
  -i test/labelme/init.db \
  importLabelmeObjects --annotations_dir test/labelme/w55-e04-objects1 \
  --keep_original_object_name --polygon_name objects1

# Import Pascal.
./shuffler.py --rootdir '.' \
  -o '/tmp/pascal.db' \
  importPascalVoc2012 \
  --images_dir ${VOC2012}/JPEGImages \
  --detection_dir ${VOC2012}/Annotations \
  --segmentation_dir ${VOC2012}/SegmentationClass

# Import BDD.
./shuffler.py  --rootdir '.' \
  -o '/tmp/bdd100k_train.db' \
  importBdd \
  --images_dir ${BDD}/bdd100k/images/100k/train \
  --detection_json ${BDD}/bdd100k/labels/bdd100k_labels_images_train.json
```

#### <a name="info">Info

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
  plotHistogram \
    'SELECT value FROM objects 
     INNER JOIN properties p1 ON objects.objectid=p1.objectid 
     WHERE name="car" AND key="yaw"' \
  --xlabel yaw --display

# Plot a "strip" plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  plotStrip \
    'SELECT p1.value, p2.value FROM objects 
     INNER JOIN properties p1 ON objects.objectid=p1.objectid 
     INNER JOIN properties p2 ON p1.objectid=p2.objectid 
     WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display

# Plot a scatter plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/databases/micro1_v4.db' \
  plotScatter \
    'SELECT p1.value, p2.value FROM objects 
     INNER JOIN properties p1 ON objects.objectid=p1.objectid 
     INNER JOIN properties p2 ON p1.objectid=p2.objectid 
     WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display
```

#### <a name="gui">GUI

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

#### <a name="filter">Filtering

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
  filterObjectsSQL \
    'SELECT objects.objectid FROM objects ' \
    'INNER JOIN properties ON objects.objectid=properties.objectid ' \
    'WHERE properties.value="blue" AND objects.score > 0.8'
# or
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  filterObjectsSQL \
    'SELECT objects.objectid FROM objects ' \
    'INNER JOIN properties p1 ON objects.objectid=p1.objectid ' \
    'INNER JOIN properties p2 ON objects.objectid=p2.objectid ' \
    'WHERE p1.value="blue" AND p2.key="pitch"'
```

#### <a name="modify">Modifications

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
  moveMedia --image_path '/tmp/images' --mask_path '/tmp/masks' --where_image 'imagefile LIKE "cars/images/%"'

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
  polygonsToMask --media "pictures" --mask_path 'cars/mask_polygons' \
    --skip_empty_masks --substitute_with_box
```

#### <a name="evaluate">Evaluation of ML tasks
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

#### <a name="write">Write a new image directory / video

```bash
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  writeImages --media 'pictures' --image_path 'cars/exported' --mask_alpha 0.5 --with_imageid

./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  cropObjects --media 'pictures' --image_path 'cars/exported' --mask_path 'cars/exportedmask' \
  --target_width 64 --target_height 64
```