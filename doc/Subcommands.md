# Commands

## List

| Sub-command | Description |
| --- | --- |
| __Import__ | |
| `importLabelme` | Import LabelMe annotations into the database. |
| `importLabelmeObjects` | Import LabelMe annotations of objects. For each `objectid` in the database, will look for an annotation in the form `objectid.xml`. |
| `importKitti` | Import KITTI annotations into the database. |
| `importPascalVoc2012` | Import annotations in PASCAL VOC 2012 format into the database. |
| `importBdd` | Import BDD annotations into a db (the format after 08-28-2018). Both image-level and object-level attributes are written to the "properties" table. "manualShape" and "manualAttributes" are ignored. Objects with open polygons are ignored. |
| `importDetrac` | Import DETRAC annotations into the database. |
| __Filter__ | |
| `filterEmptyImages` | Delete images without objects. |
| `filterImagesOfAnotherDb` | Delete images from the database that are not in the provided reference database. |
| `filterImagesSQL` | Delete images (and their objects) based on the `where` argument, that acts as a WHERE clause for SQL. |
| `filterObjectsAtBorder` | Delete bounding boxes closer than `border_thresh` to image border. |
| `filterObjectsByIntersection` | Delete objects that have a high Intersection over Union (IoU) with other objects in the same image. |
| `filterObjectsByName` | Delete objects with names not from the list. |
| `filterObjectsByScore` | Delete all objects that have score less than `score_threshold`. |
| `filterObjectsSQL` | Delete objects based on the SQL `where` argument, that acts as a WHERE clause for SQL. |
| __Modify__ | |
| `sql` | Run provided SQL commands. |
| `addVideo` | Import frames from a video into the database. |
| `addPictures` | Import picture files into the database. |
| `writeMedia` | Export images as a directory with pictures or as a video, and change the `imagefile` and `maskfile` entries to match the new recordings. |
| `cropObjects` | Crops object patches to pictures or video and saves their info as a db. All `imagefile` and `maskfile` entries in the database are deleted and path to the cropped images are recorded. |
| `headImages` | Keep the first N image entries. |
| `tailImages` | Keep the last N image entries. |
| `expandBoxes` | Expand bounding box in the four directions. |
| `moveMedia` | Change images and masks relative paths. |
| `addDb` | Adds info from `db_file` to the current open database. Objects can be merged. Duplicate image entries are ignore, but all objects associated with the image duplicates are added. |
| `splitDb` | Split the database into several sets (randomly or sequentially.) |
| `mergeObjectDuplicates` | Merge objects with identical fields `imagefile, x1, y1, width, height`. |
| `renameObjects` | Map object names. Can be used to make an imported dataset compatible with the database. |
| `repaintMask` | Repaint specific colors in masks into different colors. |
| `polygonsToMask` | Convert polygons of an object into a mask, and write it as \texttt{maskfile}. If there are polygon entries with different names, consider them as different polygons. Masks from each of these polygons are summed up and normalized to their number. The result is a black-and-white mask when there is only one polygon, and a grayscale mask when there are multiple polygons. |
| __Info__ | |
| `printInfo` | Print summary of the database. |
| `dumpDb` | Print tables of the database. |
| `plotHistogram` | Get a histogram plot of a field in the database. |
| `plotScatter` | Get a scatter plot of two fields in the database. |
| `plotStrip` | Get a "strip" plot of two fields in the database. |
| __GUI__ | |
| `examineImages` | Loop through images. Possibly, assign names to images. |
| `examineObjects` | Loop through objects. |
| `examineMatches` | Loop through matches. |
| `labelObjectsProperty` | Loop through objects and manually label them, i.e. assign the value of a property. |
| `labelMatches` | Loop through image pairs and label matching objects on the two images of each pair. |
| __Evaluate__ | |
| `evaluateDetection` | Evaluate bounding boxes in the current database w.r.t. a ground truth database. |
| `evaluateSegmentationIoU` | Evaluate masks in the current database w.r.t. a ground truth database in terms of Intersection over Union (IoU). |
| `evaluateBinarySegmentation` | Evaluate masks segmentation ROC curve w.r.t. a ground truth database. Ground truth values must be 0 for background, 255 for foreground, and the rest for dontcare. |


## Examples

### <a name="import">Imports
 
[Import](#import) most common computer vision datasets. Currently support formats of [KITTI](http://www.cvlibs.net/datasets/kitti), [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd), and [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC), and LabelMe. The list is growing, pull requests are welcome.

```bash
# Import KITTI segmentation
./shuffler.py --rootdir ${KITTI_DIR} \
  -o '/tmp/kitti_segm.db' \
  importKitti \
  --images_dir=${KITTI_DIR}/data_semantics/training/image_2  \
  --segmentation_dir=${KITTI_DIR}/data_semantics/training/instance

# Import KITTI detection (does not share images with segmentation)
./shuffler.py --rootdir ${KITTI_DIR} \
  -o '/tmp/kitti_det.db' \
  importKitti \
  --images_dir=${KITTI_DIR}/data_object_image_2/training/image_2  \
  --detection_dir=${KITTI_DIR}/data_object_image_2/training/label_2

# Import LabelMe.
# First make a database from images, then import labelme annotations.
./shuffler.py --rootdir '.' \
  -o '/tmp/labelme.db' \
  addMedia --image_pattern ${LABELME_IMAGE_DIR}'/*.jpg' \| \
  importLabelme --annotations_dir ${LABELME_XML_DIR}

# Import Pascal.
./shuffler.py --rootdir '.' \
  -o '/tmp/pascal.db' \
  importPascalVoc2012 --pascal_dir ${VOC2012_DIR} --segmentation_class

# Import BDD.
./shuffler.py  --rootdir '.' \
  -o '/tmp/bdd100k_train.db' \
  importBdd \
  --images_dir ${BDD_DIR}/bdd100k/images/100k/train \
  --detection_json ${BDD_DIR}/bdd100k/labels/bdd100k_labels_images_train.json

# Import CityScapes.
./shuffler.py  --rootdir '.' \
  -o '/tmp/cityscapes_trainval_gtfine.db' \
  importCityscapes \
  --cityscapes_dir ${CITYSCAPES_DIR} \
  --split train val --type "gtFine" --mask_type labelIds
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
  --in_db_file 'testdata/cars/micro1_v4.db' \
  examineImages

# Examine objects.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  examineObjects

# Examine matches.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  examineMatches
```

#### <a name="filter">Filtering

```bash
# Filter images that are present another .
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterImagesOfAnotherDb --ref_db_file 'testdata/cars/micro1_v4_singleim.db'

# Filter objects at image border.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterObjectsAtBorder --with_display

# Filter objects that intersect other objects too much.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterObjectsByIntersection --with_display

# Filter objects that have certain names.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterObjectsByName --good_names 'car' 'bus'

# Filter objects that have low score.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterObjectsByScore --score_threshold 0.7

# Filter objects with an SQL query
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  filterObjectsSQL \
    'SELECT objects.objectid FROM objects ' \
    'INNER JOIN properties ON objects.objectid=properties.objectid ' \
    'WHERE properties.value="blue" AND objects.score > 0.8'
# or
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
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
  --in_db_file 'testdata/cars/micro1_v4.db' \
  addVideo --image_video_path 'testdata/moon/images.avi' --mask_video_path 'testdata/moon/masks.avi'

# Add image and mask pictures.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  addPictures --image_pattern 'testdata/moon/images/*.jpg' --mask_pattern 'testdata/moon/masks/*.png'

# Discard all, but the first N images.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  headImages -n 2

# Discard all, but the last N images.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  tailImages -n 2

# Expand bounding boxes
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  expandBoxes --expand_perc 0.2 --with_display
# or to try match the target_ratio
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  expandBoxes --expand_perc 0.2 --target_ratio 0.75 --with_display

# To move the directory of images or masks
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  moveMedia --image_path '/tmp/images' --mask_path '/tmp/masks' --where_image 'imagefile LIKE "cars/images/%"'

# Add a new database (follow this one with mergeObjectDuplicates).
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  addDb --db_file 'testdata/cars/micro1_v4_singleim.db'

# Merge multiple objects that have the same ROI 
# For example, when objects were collected from different annotators and then merged via addDb.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  mergeObjectDuplicates

# Split into several databases.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  splitDb --out_dir '.' --out_names 'test' 'train' --out_fractions 0.3 0.7 --shuffle

# Masks from polygons.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  polygonsToMask --media "pictures" --mask_path 'cars/mask_polygons' \
    --skip_empty_masks --substitute_with_box
```

#### <a name="evaluate">Evaluation of ML tasks
```bash
# Evaluate semantic segmentation.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  evaluateSegmentationIoU --gt_db_file 'testdata/cars/micro1_v4_polygons.db' \
    --gt_mapping_dict '{0: "background", 255: "car"}' --out_dir 'test/testIoU'

# Evaluate single-class detection (background vs foreground), 
# where masks of current db are grayscale probabilities.
./shuffler.py --rootdir 'test' --in_db_file 'testdata/cars/micro1_v4.db' \
  evaluateBinarySegmentation --gt_db_file 'testdata/cars/micro1_v4_polygons.db'

# Evaluate object detection.
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  evaluateDetection --gt_db_file 'testdata/cars/micro1_v4_singleim.db'
```

#### <a name="write">Write a new image directory / video

```bash
./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  writeImages --media 'pictures' --image_path 'cars/exported' --mask_alpha 0.5 --with_imageid

./shuffler.py --rootdir 'test' \
  --in_db_file 'testdata/cars/micro1_v4.db' \
  cropObjects --media 'pictures' --image_path 'cars/exported' --mask_path 'cars/exportedmask' \
  --target_width 64 --target_height 64
```
