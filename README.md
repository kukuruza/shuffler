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

- [Import](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#import) most common computer vision datasets. The list of supported datasets is growing.
- [Aggregate information](#info) about a dataset. Print basic statistics, plot histograms, and scatter plots.
- [GUI](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#gui) lets a user to manually loop through a dataset, visualize, modify, and delete entries.
- [Filter](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#filter) annotations, e.g. small objects, objects at image boundary, or objects without a color.
- [Modify](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#modify) a dataset, e.g. increase bounding boxes by 20%, split a dataset into "train" and "test" subsets
- [Evaluate](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#evaluate). Given ground truth and predictions, evaluate performance of object detection or semantic segmentation.
- Export. We provide a [PyTorch Dataset class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class) to directly load data from PyTorch. I plan to implement [Keras Dataset class](https://keras.io/utils/#sequence) and export to popular formats such as PASCAL.

The toolbox supports datasets consisting of 1) images and masks, 2) objects annotated with masks, polygons, and bounding boxes, and 3) matches between objects. It stores annotations as a SQL database of its custom format. This database can be viewed and edited manually with any SQL viewer.

Example:

```bash
./shuffler.py -o myPascal.db \
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


## Installation 

#### Using conda

Shuffler requires Python3. The installation instructions assume Conda package management system.

Install dependencies:

```bash
conda create -n shuffler python=3
conda activate shuffler

conda install -y -c conda-forge ffmpeg=4.0
conda install -y imageio matplotlib lxml simplejson progressbar2 pillow scipy opencv=3

# If desired, add support for plotting commands
conda install -y pandas seaborn

# If desired, add support for unit tests
conda install -y nose scikit-image
```

Clone this project:

```bash
git clone https://github.com/kukuruza/shuffler
cd shuffler
```

The basic installation is okay if the following command does not break with an import error:

```bash
./shuffler.py printInfo
```

### Install only interface to Keras, Pytorch, and DatasetWriter

While `shuffler.py` tool requires Python 3, the Keras generators, Pytorch datasets, and Shuffler Dataset Writer can be run in Python 2 or Python 3.

```bash
conda install -y -c conda-forge ffmpeg=4.0
conda install -y imageio progressbar2 pillow numpy opencv=3
```




## Gentle introduction

### Chaining commands

Sub-commands can be chained via the special symbol "\|" (here, the backslash escapes the following vertical bar from a Unix shell.)

```bash
./shuffler.py --rootdir 'test' --in_db_file 'test/cars/micro1_v4.db' \
  addVideo --image_video_path 'test/moon/images.avi' --mask_video_path 'test/moon/masks.avi' \| \
  printInfo \| \
  moveMedia --image_path 'test/cars/images' --where_image 'imagefile LIKE "cars/images/%"' \| \
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
  mergeObjectDuplicates \| \
  polygonsToMask --mask_pictures_dir 'test/labelme/mask_polygons' --skip_empty_masks \| \
  dumpDb --tables objects polygons \| \
  examineImages --mask_aside
```


## Examples of getting information about a dataset with standard SQLite.
```bash
# Print out names of objects and their count.
sqlite3 my_dataset.db "SELECT name, COUNT(1) FROM objects GROUP BY name"
```

## Testing code

Most of the backend and utilities are covered in unit tests. To run all tests, run:

```bash
cd test
python3 -m "nose"
```
