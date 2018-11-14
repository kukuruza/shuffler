# shuffler

Toolbox for manipulating image annotations in computer vision.

- [Motivation](#motivation)
- [Functionality](#functionality)
- [Example use cases](#example-use-cases)
- [Installation](#installation)
- [Gentle introduction](#gentle-introduction)

![data preparation pipeline](fig/data-preparation-pipeline.png)

## Motivation

Experts in computer vision train machine learning models to tackle practical problems, such as detecting vehicles in the autonomous car scenario or find faces in Facebook pictures. In order to train a model, researchers either use public datasets of annotated images or collect their own. In the process of fighting for better model performance, a researcher may want to change or filter image annotations, or to add another public dataset. Currently, each small group of researchers writes their own sripts to load, change, and save annotations. As the number of experiments grows, these custom scripts become more and more difficult to maintain. **Shuffler** eliminates the need for custom scripts by providing a multipurpose tool to import, modify, visualize, export, and evaluate annotations for common computer vision tasks.

## Functionality

Shuffler is a command line tool. It takes a dataset in one of the formats on inputs, performs a number of *operations*, and then records the output. Operations fall under these categories:

- [Import](#import) most common computer vision datasets. The list of supported datasets is growing.
- [Aggregate information](#info) about a dataset. Print basic statistics, plot histograms, and scatter plots.
- [GUI](#gui) lets a user to manually loop through a dataset, visualize, modify, and delete entries.
- [Filter](#filter) annotations, e.g. small objects, objects at image boundary, or objects without a color.
- [Modify](#modify) a dataset, e.g. increase bounding boxes by 20%, split a dataset into "train" and "test" subsets
- [Evaluate](#evaluate). Given ground truth and predictions, evaluate performance of object detection or semantic segmentation.
- Export. We provide a [PyTorch Dataset class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class) to directly load data from PyTorch. I plan to implement [Keras Dataset class](https://keras.io/utils/#sequence) and export to popular formats such as PASCAL.

The toolbox supports datasets consisting of 1) images and masks, 2) objects annotated with masks, polygons, and bounding boxes, and 3) matches between objects. It stores annotations as a SQL database of its custom format. This database can be viewed and edited manually with any SQL viewer.

Example:

```bash
./shuffler -o myPascal.db \
  importPascalVoc2012 ${VOC_DIR} --annotations \| filterObjectsSQL "SELECT objectid WHERE width < 20"
```

In this example, we import PASCAL VOC 2012 dataset from `${VOC_DIR}`, remove small objects, and save the annotations as an SQLite database `myPascal.db`. Later we may choose to export it back to the PASCAL format or to load data from `myPascal.db` to PyTorch directly. 


## Example use cases

#### Combine [KITTI](http://www.cvlibs.net/datasets/kitti), [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd), and [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC) datasets into one ([link to code](#combine-datasets).)

A user works on object detection tasks for the autonomous car scenario, and would like to use as many annotated images as possible. In particular, they aim to combine certain classes from the datasets KITTI, BDD, and PASCAL VOC 2012. Then the combined dataset should be exported to a TF-friendly format.

#### Import annotations from [LabelMe](http://labelme.csail.mit.edu/Release3.0). Each image is labelled by multiple annotators ([link to code](#import-from-labelme))

A user has collected a dataset of images with objects. Images were handed out to annotators who use LabelMeAnootationTool. Each image was annotated with polygons by multiple annotators for the purposes of cross-validation. A user would like to to 1) import labels from all annotators, 2) merge polygons corresponding to the same object, 3) make black-gray-white image masks, where the gray area marks the inconsistency among annotators.

#### Train object detection with only big objects. Then evaluate properly.

We have a dataset with objects given as bounding boxes. We would like to remove objects on image boundary, expand bounding boxes by 10% for better training, remove objects of all types except "car", "bus", and "truck", and to remove objects smaller than 30 pixels wide. We would lile to use that subset for training.

In the previous use case we removed some objects for our object detection training task. Now we want to evaluate the trained model. We expect our model to detect only big objects, but we don't want to count it as a false positive if it detects a tiny object either.

#### Evaluate results of semantic segmentation

A neural network was trained to perform a semantic segmentation of images. We have a directory with ground truth masks and a directory with predicted masks. We would like to 1) evaluate the results, 2) write a video with images and their predicted segmentation masks side-by-side.

#### Write a dataset with image croppings of individual objects

We have images with objects. Images have masks with those objects. We would like to crop out objects with name "car" bigger than 32 pixels wide, stretch the crops to 64x64 pixels and write a new dataset of images (and the correspodning masks)




## Gentle introduction





## Installation 

#### Using conda

Shuffler requires Python3. The installation instructions assume Conda package management system.

```bash
# If desired, add support for datasets stored as video (needs to go first).
conda install -c conda-forge ffmpeg=4.0

conda install imageio matplotlib lxml simplejson progressbar2 Pillow scipy
conda install opencv=3.4.2  # Require opencv3.

# If desired, add support for plotting commands
conda install pandas seaborn

# If desired, add support for unit tests
conda install nose scikit-image
```


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
  --in_db_file 'test/cars/micro1_v4.db' \
  printInfo --objects_by_image

# Print several tables of a database
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  dumpDb --tables objects properties

# Plot a histogram of value of field "yaw" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  plotObjectsHistogram \
  --sql_query 'SELECT value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               WHERE name="car" AND key="yaw"' \
  --xlabel yaw --display

# Plot a "strip" plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  plotObjectsStrip \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
               INNER JOIN properties p1 ON objects.objectid=p1.objectid 
               INNER JOIN properties p2 ON p1.objectid=p2.objectid 
               WHERE name="car" AND p1.key="yaw" AND p2.key="pitch"' \
  --xlabel yaw --ylabel pitch --display

# Plot a scatter plot of values of field "yaw" and "pitch" of objects named "car".
./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  plotObjectsScatter \
  --sql_query 'SELECT p1.value, p2.value FROM objects 
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
  polygonsToMask --media='pictures' --mask_path 'cars/mask_polygons' --skip_empty_masks
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
  writeMedia --media 'pictures' --image_path '/tmp/shuffler/cars/images' \
  --mask_alpha 0.5 --with_imageid --overwrite

./shuffler.py --rootdir 'test' \
  --in_db_file 'test/cars/micro1_v4.db' \
  cropObjects --media 'pictures' --image_path '/tmp/shuffler/cars/imagecrops' \
  --mask_path '/tmp/shuffler/cars/maskcrops' \
  --target_width 64 --target_height 64 --overwrite
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
  polygonsToMask --media='pictures' --mask_path 'cars/mask_polygons' --skip_empty_masks \| \
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
