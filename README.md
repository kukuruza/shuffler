# Shuffler

Shuffler is a Python library for data engineering in computer vision. It simplifies building, maintaining, and inspection of datasets for machine learning.

For example, you are building a dataset to train a vehicle classifier. You may start by downloading the public [BDD dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd). Then you (1) remove annotations of everything but vehicles, (2) filter out all tiny vehicles, (3) expand bounding boxes by 20% to include some context, (4) crop out the bounding boxes, (5) save annotations in the [ImageNet format](https://www.tensorflow.org/datasets/catalog/imagenet2012) to be further fed to [TensorFlow](https://www.tensorflow.org/). Shuffler allows to do that by running a single command in the terminal ([see use case #1](#crop-vehicles-from-bdd)).

![crop-from-bdd](https://habrastorage.org/webt/ku/vt/1g/kuvt1g6zs42-st68cpyfqm9uqd0.gif)

## Table of contents

- [Why data engineering](#why-data-engineering)
- [Why Shuffler](#why-shuffler)
- [Author](#author)
- [Installation](#installation)
- [Getting started](#getting-started)
  - [A simple example](#a-simple-example)
  - [Operations](#operations)
  - [Chaining operations](#chaining-operations)
  - [Pytorch and Keras API](#pytorch-and-keras-api)
- [FAQ](#faq)
  - [What can I do with Shuffler](#what-can-i-do-with-shuffler)
  - [I just want to convert one dataset format to another](#i-just-want-to-convert-one-dataset-format-to-another)
  - [I want to use Shuffler with a deep learning framework](#i-want-to-use-shuffler-with-a-deep-learning-framework)
  - [What ML tasks does Shuffler support](#what-ml-tasks-does-shuffler-support)
  - [How does Shuffler compare to package N](#how-does-shuffler-compare-to-package-n)
  - [Is there dataset versioning](#is-there-dataset-versioning)
  - [How is a dataset stored](#how-is-a-dataset-stored)
- [Example use cases](#example-use-cases)
- [SQL schema](#sql-schema)
- [Contributing](#contributing)
- [Citing](#citing)

--------------------------------------

## Why data engineering

Data engineering for machine learning means building and maintaining datasets.

Research groups in academia compare their algorithms on publicly available datasets, such as [KITTI](http://www.cvlibs.net/datasets/kitti). In order to allow comparison, public datasets must be static. On the other hand, a data scientist in industry enhances both algorithms AND datasets in order to achieve the best performance on a task. That includes collecting data, cleaning data, and fitting data for a task. Some even treat [data as code](https://towardsdatascience.com/data-as-code-principles-what-it-is-and-why-now-aaf1e24fa732). This is data engineering.

You may need a data engineering package if you find yourself writing multiple scripts with of lot of boilerplate code for simple operations with data, if your scripts are becoming [write-only code](https://encyclopedia2.thefreedictionary.com/write-only+code), if you have multiple modifications of the same dataset, e.g. folders named "KITTI", "KITTI-only-vans", "KITTI-inspected", etc.

## Why Shuffler

- Supports of the most [common computer vision tasks](#what-ml-tasks-does-shuffler-support).
- Shuffler is easy to [set up and use](#getting-started), and suits datasets of up to millions of annotated images.
- Shuffler [allows](#operations) to import/export popular formats, filter and modify annotations, evaluate ML results, inspect and visualize a dataset.
- All metadata (annotations and paths to images/videos) is stored [in a single SQLite file](#sql-schema), which can be manually inspected and modified.
- Shuffler provides an [API for Pytorch and Keras](#pytorch-and-keras-api).
- Written with practical usage in mind. Shuffler has been evolving according to the needs of real projects.
- [Easily extendable](doc/Extending-functionality.md) for your needs.


## Author

[Evgeny Toropov](https://www.linkedin.com/in/evgeny-toropov-9bb14210b/)

--------------------------------------

## Installation

Shuffler requires Python3. The installation instructions assume Conda package management system.

Install dependencies:

```bash
conda install -c conda-forge imageio ffmpeg=4 opencv matplotlib
conda install lxml simplejson progressbar2 pillow scipy
conda install pandas seaborn  # If desired, add support for plotting commands
```

Clone this project:

```bash
git clone https://github.com/kukuruza/shuffler
```

To test the installation, run the following command. The installation succeeded if an image opens up. Press Esc to close the window.

```bash
cd shuffler
python -m shuffler -i 'testdata/cars/micro1_v5.db' --rootdir 'testdata/cars' examineImages
```


## Getting started

### A simple example

Shuffler is a command line tool. It chains operations, such as `importKitti` to import a dataset from [KITTI](http://www.cvlibs.net/datasets/kitti) format and `exportCoco` to export it in [COCO format](https://cocodataset.org/#home).

```bash
python -m shuffler \
  importKitti --images_dir ${IMAGES_DIR} --detection_dir ${OBJECT_LABELS_DIR} '|' \
  exportCoco --coco_dir ${OUTPUT_DIR} --subset 'train'
```

### Operations

`importKitti` and `exportCoco` above are examples of operations. There are over 60 operations that fall under the following broad categories:

- [Import/export](doc/Operations.md#import) most common computer vision datasets.
- Aggregate information about a dataset. Print statistics, plot histograms and scatter plots.
- [GUI](doc/Operations.md#gui) to manually loop through a dataset, visualize, modify, and delete entries.
- [Filter](doc/Operations.md#filter) annotations, e.g. small objects, objects at image boundary, objects without a color, etc.
- [Modify](doc/Operations.md#modify) a dataset, e.g. increase bounding boxes by 20%, split a dataset into "train" and "test" subsets, etc.
- [Evaluate](doc/Operations.md#evaluate) the performance of an object detection or semantic segmentation task, given the ground truth and predictions.


### Chaining operations

Sub-commands can be chained via the vertical bar `|`, similar to pipes in Unix. The vertical bar should be quoted or escaped. Using single quotes `'|'` works in Windows, Linux, and Mac. Alternatively, in Unix, you can escape the vertical bar as `\|`.

The next example (1) opens a database, (2) converts polygon labels to pixel-by-pixel image masks (3) adds more images with their masks to the database, and (4) prints summary.

```bash
python -m shuffler --rootdir 'testdata/cars' -i 'testdata/cars/micro1_v5.db' \
  polygonsToMask --media='pictures' --mask_path 'testdata/cars/mask_polygons' '|' \
  addPictures --image_pattern 'testdata/moon/images/*.jpg' --mask_pattern 'testdata/moon/masks/*.png' '|' \
  examineImages --mask_alpha 0.5 \
  printInfo
```

### Pytorch and Keras API

Shuffler has an interface to Pytorch: classes [ImageDataset and ObjectDataset](interface/pytorch/datasets.py) implement `torch.utils.data.Dataset`.
A [demo](interface/pytorch/datasets_demo.py) provides an example of using a Shuffler database as a Dataset in Pytorch inference.

Shuffler also has an interface to Keras: classes [Imaginterface/keras/generetors.py) implement `keras.utils.Sequence`.
A [demo](interface/keras/generators_demo.py) provides an example of using a Shuffler database as a Generator in Keras inference.

Alternatively, data can be exported to one of the popular formats, e.g. [COCO](https://cocodataset.org/#home), if your deep learning project already has a loader for it.

--------------------------------------

## FAQ

### What can I do with Shuffler
Shuffler is for inspecting and modifying your datasets. Check out some [use cases](#example-use-cases).

### I just want to convert one dataset format to another
You can convert one format to another, like in the [example below](#a-simple-example). Check out the [dataset IO] tutorial.

### I want to use Shuffler with a deep learning framework
Shuffler provides an [API to Pytorch and Keras](#pytorch-and-keras-api).

### What ML tasks does Shuffler support?
[Shuffler's database schema](#sql-schema) is designed to support computer vision tasks, in particular image classification, object and panoptic detection, image and instance segmentation, object tracking, object detection in video.

### How does Shuffler compare to package N?
- **Deep learning frameworks.** Shuffler prepares datasets for machine learning and has [an API to feed data into Keras and Pytorch](#pytorch-and-keras-api).
- **ML workflow management services**, such as [wandb.ai](https://wandb.ai/), allow you to tag experiments and input data, but are not designed to work with datasets. Shuffler database filenames can be used as a dataset tag in these services.
- **Data augmentation** libraries, such as [Albumentation](https://albumentations.ai/) or [Nvidia DALI](https://developer.nvidia.com/dali), let you modify data on the fly as it is fed into ML training. They can and should be used by Shuffler's [Pytorch and Keras API](#pytorch-and-keras-api).

### Is there dataset versioning?
Shuffler does not support versions inside the database SQL schema. The version can be a part of the database name, e.g. `dataset.v1.db` and `dataset.v2.db`.

### How is a dataset stored?
A dataset consists of (1) image data, stored as image and video files, and (2) metadata, stored as the SQLite database. [Shuffler's SQL schema](#sql-schema) is designed to [support popular machine learning tasks](#what-ml-tasks-does-shuffler-support) in computer vision.

--------------------------------------

## Example use cases

#### Crop vehicles from BDD

The public [BDD dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd) includes 100K images taken from a moving car with various objects annotated in each image. If a researcher wants to train a classifier between "car", "truck", and "bus", they may start by using this dataset. First, annotations of all objects except for these three classes must be filtered out. Second, the dataset annotations for tons of tiny vehicles, which would not be good for a classifier. Third, it may be beneficial to expand bounding boxes to allow for [data augmentation](https://albumentations.ai/) during training. Fourth, the remaining objects need to be cropped out. The cropped images and the annotations are saved in ImageNet format, which is easily consumable by TensorFlow. The KITTI dataset is assumed to be downloaded to directories `${IMAGES_DIR}` and `${OBJECT_LABELS_DIR}`.

```bash
python -m shuffler \
  importKitti --images_dir ${IMAGES_DIR} --detection_dir ${OBJECT_LABELS_DIR}  '|' \
  filterObjectsByName --good_names 'car' 'truck' 'bus'  '|' \
  filterObjectsSQL --sql "SELECT objectid FROM objects WHERE width < 64 OR height < 64"  '|' \
  expandObjects --expand_fraction 0.2  '|' \
  cropObjects --media 'pictures' --image_path ${NEW_CROPPED_IMAGE_PATH} --target_width 224 --target_height 224  '|' \
  exportImagenet2012 --imagenet_home ${NEW_IMAGENET_DIRECTORY} --symlink_images
```

#### Import and merge LabelMe annotations

A researcher has collected a dataset of images with cars. Images were handed out to a team of annotators. Each image was annotated with polygons by several annotators using [LabelMeAnootationTool](http://labelme.csail.mit.edu/Release3.0). The researcher 1) imports all labels, 2) merges polygons corresponding to the same car made by all annotators, 3) gets objects masks, where the gray area marks the inconsistency across annotators. See the [tutorial](doc/use-cases/merge-annotations/merge-annotations.md).

#### Combine several public datasets

A user works on object detection in the autonomous vehicle setup, and would like to use as many annotated images as possible. In particular, they aim to combine certain classes from the public datasets [KITTI](http://www.cvlibs.net/datasets/kitti), [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd), and [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC). The combined dataset is exported in [COCO](https://cocodataset.org/#home) format for training. See the [tutorial](doc/use-cases/combining-datasets/combining-datasets.md).

#### Train and evaluate object detection with big objects

We have a dataset with objects given as bounding boxes. We would like to remove objects on image boundary, expand bounding boxes by 10% for better training, remove objects of all types except "car", "bus", and "truck", and to remove objects smaller than 30 pixels wide. We would lile to use that subset for training.

In the previous use case we removed some objects for our object detection training task. Now we want to evaluate the trained model. We expect our model to detect only big objects, but we don't want to count it as a false positive if it detects a tiny object either.

#### Evaluate results of semantic segmentation

A neural network was trained to perform a semantic segmentation of images. We have a directory with ground truth masks and a directory with predicted masks. We would like to 1) evaluate the results, 2) write a video with images and their predicted segmentation masks side-by-side.

#### Write a dataset with image crops of individual objects

We have images with objects. Images have masks with those objects. We would like to crop out objects with name "car" bigger than 32 pixels wide, stretch the crops to 64x64 pixels and write a new dataset of images (and the correspodning masks)

#### Manually relabel objects

A dataset contains objects of class "car", among other classes. We would like to additionally classify cars by type for more fine-grained detection. An image annotator needs to go through all the "car" objects, and assign one of the following types to them: "passenger", "truck", "van", "bus", "taxi". See the [tutorial](doc/use-cases/manual-labelling/manual-labelling.md).

--------------------------------------

## SQL schema

Shuffler stores metadata as an SQLite database. Metadata includes image paths and annotations.

You can import some well-known formats and save them in Shuffler's format. For example, importing [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC) looks like this. We assume you have downloaded PASCAL VOC to `${VOC_DIR}`:

```bash
python -m shuffler -o 'myPascal.db' importPascalVoc2012 ${VOC_DIR} --annotations
```

You can open `myPascal.db` with any SQLite3 editor/viewer and manually inspect data entries, or run some SQL on it.

The toolbox supports datasets consisting of 1) images and masks, 2) objects annotated with masks, polygons, and bounding boxes, and 3) matches between objects. It stores annotations as a SQL database of its custom format. This database can be viewed and edited manually with any SQL viewer.


### Using 3rd party SQLite editors with Shuffler databases.

The beauty of storing annotations in a relational SQLite database is that one can use any SQL editor to explore them. For example, Linux includes the command line tool `sqlite3`.

The commands below illustrate using `sqlite3` to get some statistics and change `testdata/cars/micro1_v5.db` from this repository.

```bash
# Find the total number of images:
sqlite3 testdata/cars/micro1_v5.db 'SELECT COUNT(imagefile) FROM images'

# Find the total number of images with objects:
sqlite3 testdata/cars/micro1_v5.db 'SELECT COUNT(DISTINCT(imagefile)) FROM objects'

# Print out names of objects and their count:
sqlite3 testdata/cars/micro1_v5.db 'SELECT name, COUNT(1) FROM objects GROUP BY name'

# Print out dimensions of all objects of the class "car":
sqlite3 testdata/cars/micro1_v5.db 'SELECT width, height FROM objects WHERE name="car"'

# Change all names "car" to "vehicle".
sqlite3 testdata/cars/micro1_v5.db 'UPDATE objects SET name="vehicle" WHERE name="car"'
```

--------------------------------------

## Contributing

Please submit a pull request or open an issue with a suggestion.

## Citing

If you find this project useful for you, please consider citing:

```
@inproceedings{10.1145/3332186.3333046,
  author = {Toropov, Evgeny and Buitrago, Paola A. and Moura, Jos\'{e} M. F.},
  title = {Shuffler: A Large Scale Data Management Tool for Machine Learning in Computer Vision},
  year = {2019},
  isbn = {9781450372275},
  series = {PEARC '19}
}
```
