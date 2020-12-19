# Crash course to Shuffler

## About

The goal of this crash course is to teach a user by example how to use Shuffler
for solving practical problems.

## Basics

### A dataset is a database plus images.

A Shuffler dataset consists of a SQLite database and videos or image folders.

For example, one of the test datasets that is used in Shuffler Unit tests
consists of a database, of a folder with images, and a folder of masks:

- `testdata/cars/micro1_v4.db`
- `testdata/cars/images`
- `testdata/cars/masks`

The SQLite database (`testdata/cars/micro1_v4.db`) stores the information about
where each images is located, about objects in each image, object attributes,
their ROI, and objects in the other images that they can be matched against.

The information about images is stored in the table `images` of the database.
Let's look what does this table have in `testdata/cars/micro1_v4.db`.

### Looking at `images` table.

All the work with Shuffler is currently done via a terminal (Mac, Linux, Windows
all work.) Open your terminal, navigate to the root directory of the shuffler
repo, and type:

```bash
sqlite3 testdata/cars/micro1_v4.db "SELECT * FROM images"
```

Note that we called `sqlite3` (in the Shuffler requirements list), which has
nothing to do with Shuffler. It allows to view and edit Sqlite3 databases,
including the ones used by Shuffler. In essence, a Shuffler database is really
a certain schema of the SQLite database.

This command should yield a response:

```
images/000000.jpg | 800 | 700 | masks/000000.png | 2018-09-24 12:22:48.534685 | |
images/000001.jpg | 800 | 700 | masks/000001.png | 2018-09-24 12:22:48.562608 | |
images/000002.jpg | 800 | 700 | masks/000002.png | 2018-09-24 12:22:48.581214 | |
```

The first column corresponds to the field `imagefile` from the table `images`.
One can see that the dataset consists of 3 files, all of them in folder
`images/`. You may have noticed that it is not an absolute path in a file
system, but a relative path. It is relative to a so called root directory.
In this case, the root directory is `testdata/cars` inside the Shuffler repo.

The other columns of the database will be discussed later.

### Displaying images.

It's time to look at an image via the `shuffler.py` command line tool.
NOTE: this will only work if you have a display, as opposed to e.g. working in
a terminal via ssh.
Navigate to the root directory of the Shuffler repo, and type:

```bash
./shuffler.py -i "testdata/cars/micro1_v4.db" --rootdir "testdata/cars" examineImages
```

You should see the following image pop up in a new window on your screen.
If you see an error in the terminal output, first check your dependencies.
Using your mouse, click somewhere in the window with the image.
Now press "-" and "=" on your keyboard to loop through images in the dataset.
If you happened to switch to a different application, you need to click on
the window with the image to continue navigation.
When done, press "Esc".

![test cars micro db](Crash-course/cars-micro-image1-1500x500.png)

You have just run the command line tool `./shuffler.py`. It loaded the database
`testdata/cars/micro1_v4.db`, and ran its sub-command `examineImages`.
Note that the `rootdir` argument was specified to let Shuffler know the relative
directory of images, in this case, folder `testdata/cars`. Either the absolute or
relative path can be specified as `rootdir`.

Shuffler sub-commands, including `examineImages` usually come with their own
arguments. Run the following command to see not only the images, but also
the labelled objects:

```bash
./shuffler.py -i "testdata/cars/micro1_v4.db" --rootdir "testdata/cars" \
  examineImages --with_objects
```

![test cars micro db with objects](Crash-course/cars-micro-image2-1500x500.png)

You may notice that the first image has a labelled car, the second has a car
and a bus, and the last one does not have any objects.

##### Exercise.

Copy the folder with images `testdata/cars/images` and a folder with their masks
`testdata/cars/masks` to different location, say, your Desktop.
Now examine images using the previous command but specify the new
`rootdir`.

<details>
  <summary>Answer</summary>

In my case the new image folder is at `$HOME/Desktop`.

```bash
./shuffler.py -i "testdata/cars/micro1_v4.db" --rootdir $HOME"/Desktop" examineImages
```

</details>


<!-- ##### Objects table. -->




