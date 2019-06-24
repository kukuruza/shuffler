# Combining data from multiple datasets to train an object detector.

In this tutorial, we are going to make a dataset for the autonomous industry from multiple available public datasets.
In particular, we will prepare data for training an object detector. The training data for such a task is a pool of pairs {image, bounding boxes}. Here, bounding boxes are essentially rectangles around each object of interest in the image.
The common classes of interest for self-driving are pedestrians, vehicles, bikes, traffic lights, traffic signs.

There exist several public datasets for the self-driving scenario. We will consider PASCAL VOC, KITTI, Cityscapes, and BDD. 
Each of them contains data in its own format. 
They contain annotated objects of various classes, beyond the ones we are interested in, such as "bird" or "sky".
Moreover, some datasets have class "car", some -- "Car", and some distinguish its subclasses.

Our goal is to combine all those objects, filter out unnecessary classes, and rename the classes so that we end up with the following categories:

- pedestrian
- vehicle (all types of cars)
- cyclist (bycicle plus the person(s) on top)
- motobiker (motorbike plus the person(s) on top)
- traffic light
- traffic sign
- dont care (ambigouos things that may confuse an algorithm)


### Import each dataset. 

The first step is to create a database for each of the datasets. The reader is referred to the [Import Section](https://github.com/kukuruza/shuffler/blob/master/doc/Subcommands.md#import).


### Combine datasets.

The following command combines all four datasets into one, called `combined.db`. We assume that the environmental variables such as `CITYSCAPES_DIR` were defined in the previous step. Argument `db_rootdir` specifies the directory image paths are considered to be relative to.

```
./shuffler.py -o combined.db \
  addDb --db_file $CITYSCAPES_DIR/cityscrapes_trainval_gtfine.db --db_rootdir $CITYSCAPES_DIR \| \
  addDb --db_file $KITTI_DIR/kitti_detection.db --db_rootdir $KITTI_DIR \| \
  addDb --db_file $PASCAL_DIR/pascal.db --db_rootdir $PASCAL_DIR \| \
  addDb --db_file $BDD_DIR/bdd100k_detection_train.db --db_rootdir $BDD_DIR
```


### Merge bikes/motorbikes with their riders

This is where we are faced again with incompaticle annotations. BDD and Cityscapes have class "rider" that refers to a person on top of the bike or a motorcycle. KITTI has the type "Cyclist". We already merged bicycles and motobikes together, now we are going to merge riders with their bikes.

In order to do that, we will merge bounding boxes of class "bicycle"/"bike" and of class "rider" when they intersect. The same goes for classes "motocycle"/"motor".

```
Intersect !!!
```


### Get objects to the common names.

All different types of cars and names for them are merged into one single category "vehicle". The same apllies to (motor)bikes. "Dont care" class combines things that are either specific to a dataset (i.e. "ego vehicle") or may confuse a detection algorithm if taken separately.

This step can be performed with the command-line tool `sqlite3`.

```
sqlite3 combined.db '
  UPDATE objects SET name="vehicle" WHERE name IN ("car", "bus", "trailer", "truck", "train", "caravan", "Truck", "Car", "Tram"); 
  UPDATE objects SET name="pedestrian" WHERE name IN ("person", "Pedestrian", "Person_sitting"); 
  UPDATE objects SET name="cyclist" WHERE name IN ("Cyclist"); 
  UPDATE objects SET name="dont care" WHERE name IN ("DontCare", "ridergroup", "truckgroup", "bicyclegroup", "persongroup", "ego vehicle", "out of roi", "rectification border");
'
```

### Remove unused names.

We have to do this via Shuffler, not in `sqlite3` tool, because deleting an object also requires deleting its properties and polygons. Shuffler takes care of that.

```
./shuffler.py -i combined.db -o combined.db \
  filterObjectsByName --good_names "vehicle" "motorbiker" "cyclist" "traffic sign" "traffic light" "dont care"
```

### Remove small objects.

BDD has an annoying feature of labelling the objects that are so far away and small that it's not even clear why they are labelled like that. In any case, many detection algorithms do not work with tiny objects. Let us convert them into "dont care" class.

```
sqlite3 combined.db 'UPDATE objects SET name="dont care" WHERE width < 10 OR height < 10'
```


### Clean up

Finally, we can optimize the performance and size of the database by cleaning up after deleting so many objects.

```
sqlite3 combined.db "VACUUM;"
```

### Inspect

This should really be done throughout the process. At least, when you decide for yourself which objects to delete, which objects to rename, etc. However, let us put it 


Inspect !!!

Plots !!!

## Aftermath

Here, we did a lot of work
