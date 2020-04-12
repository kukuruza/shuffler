# Crash course to Shuffler

## About

The goal of this crash course is to give a user the understanding of what
Shuffler is all about and what it can be used for. The "what is all about" part
is arguably difficult to explain in bare words, so why not to learn from
examples?

## Basics

##### A dataset is a database plus images.

A Shuffler dataset consists of a SQLite database and videos or image folders.

For example, one of the test datasets that is used in Shuffler Unit tests
consists of a database, of a folder with images, and a folder of masks:

- `test/cars/micro1_v4.db`
- `test/cars/images`
- `test/cars/masks`

The SQLite database (`test/cars/micro1_v4.db`) stores the information about
where each images is located, about objects in each image, object attributes,
their ROI, and objects in the other images that they can be matched against.

The information about images is stored in the table `images` of the database.
Let's look what does this table have in `test/cars/micro1_v4.db`.

##### Looking at `images` table.

All the work with Shuffle is currently done via a terminal (Mac, Linux, Windows
all work.) Open your terminal, navigate to the root directory of the shuffler
repo, and type:

```bash
sqlite3 test/cars/micro1_v4.db "SELECT * FROM images"
```

Note that we called `sqlite3` (in the Shuffler requirements list), which has
nothing to do with Shuffler. It allows to view and edit Sqlite3 databases,
including the ones used by Shuffler. In essence, a Shuffler database is really
a certain schema of the SQLite database.

This command should yield a response:

```
images/000000.jpg | 800| 700 | masks/000000.png | 2018-09-24 12:22:48.534685 | |
images/000001.jpg | 800| 700 | masks/000001.png | 2018-09-24 12:22:48.562608 | |
images/000002.jpg | 800| 700 | masks/000002.png | 2018-09-24 12:22:48.581214 | |
```

The first column corresponds to the field `imagefile` from the table `images`.
One can see that the dataset consists of 3 files, all of them in folder
`images/`. You may have noticed that it is not an absolute path in a file
system, but a relative path. It is relative to a so called root directory.
In this case, the root directory is `test/cars` inside the Shuffler repo.

The other columns of the database will be discussed later.

##### Loading images.

It's time to look at an image via the `shuffler.py` command line tool.
Navigate to the root directory of the Shuffler repo, and type:

```bash
./shuffler.py -i "test/cars/micro1_v5.db" --rootdir "test/cars" examineImages
```

You should see an image below. If you see an error, check your dependencies
and look into the issues page on Github.
You can press "-" and "=" to scroll to other images inside the dataset.
When done, press "Esc".

![test cars micro1 db](Crash-course/cars-micro-image1-1500x500.png)

You have just run the command line tool `./shuffler.py`. It loaded the database
`test/cars/micro1_v5.db`, and ran its sub-command `examineImages`.
Note that the `rootdir` argument was specified to let Shuffler know the relative
directory of images, in this case, folder `test/cars`. Either the absolute or
relative path can be specified as `rootdir`.

Shuffler sub-commands, including `examineImages` usually come with their own
arguments. Run the following command to see not only the images, but also
the labelled objects:

```bash
./shuffler.py -i "test/cars/micro1_v5.db" --rootdir "test/cars" \
  examineImages --with_objects
```

You may notice that the first image has a labelled car, the second has a car
and a bus, and the last one does not have any objects.

###### Exercise.

Copy the folder with images `test/cars/images` and a folder with their masks
`test/cars/masks` to different location, say, your Desktop.
Now examine images using the previous command but specify the new
`rootdir`.

<details>
  <summary>Answer</summary>

In my case the new image folder is at `$HOME/Desktop`.

```bash
./shuffler.py -i "test/cars/micro1_v5.db" --rootdir $HOME"/Desktop" examineImages
```

</details>


<!-- ##### Objects table. -->




