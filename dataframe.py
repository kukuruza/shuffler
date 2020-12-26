'''
This is an interface between Shuffler subcommands and ipython notebook.

It introduces Dataframe class, that is the only Shuffler interface class
for an IPython notebook. Every Shuffler subcommand is a method in the Dataframe 
class.

The workflow is expected to be like below. Refer to "dataframe_demo.ipynb" file.

# Init.
df = Dataframe()
df.load('testdata/cars/micro1_v4.db', rootdir='testdata/cars')

# Shuffler subcommands.
df.sql(sql='DELETE FROM properties')
df.printInfo()
df.displayImagesPlt(limit=4, with_objects=True, with_imagefile=True)
plt.show()

# Use objects / images outside of Shuffler.
print(len(df))
image = df[1]['image']
objects = df[1]['objects']
plt.imshow(image)

# Closing.
tmp_file_path = tempfile.NamedTemporaryFile().name
df.save(tmp_file_path)
df.close()
'''

import os
import logging
import argparse
import inspect
import sqlite3
import shutil
import tempfile
from functools import partial

from lib import subcommands
from lib.backend import backendDb
from lib.backend import backendMedia

# All the files with subcommands start with this prefix, e.g. "dbModify.py".
SUBCOMMAND_PREFIX = 'db'


def _collect_subcommands():
    '''
    Collect all subcommand functions and their parsers from subcommand modules.
    '''
    # Iterate modules (files) in subcommands package.
    all_functions = []
    for module_name in dir(subcommands):
        if not module_name.startswith(SUBCOMMAND_PREFIX):
            continue
        module = getattr(subcommands, module_name)

        functions = inspect.getmembers(module)
        functions = [f for f in functions if inspect.isfunction(f[1])]
        functions_dict = dict(functions)

        # Iterate functions in a module.
        for subparser_name in functions_dict:
            if not subparser_name.endswith('Parser'):
                continue
            subcommand_name = subparser_name[:-len('Parser')]
            if not subcommand_name in functions_dict:
                logging.error('Weird: have function %s, but not function %s',
                              subparser_name, subcommand_name)
                continue
            getattr(subcommands, module_name)

            all_functions.append(functions_dict[subcommand_name])

    return all_functions


def _make_subcommand_method(cursor, subcommand, parser, **global_kwargs):
    '''
    Given a subcommand method and its parser method, add a class method that
    would parse the input **kwargs and call the subcommand.

    This whole function is a struggle with the argparse package. In particular,
    it is a interface between **kwargs arguments that the DataFrame takes and
    the parsed arguments that subcommands take.
    '''
    def kwargs_to_argv(kwargs):
        ''' Get arguments as a dictionary, and make an input for the parser. '''
        argv = [subcommand.__name__]
        for key, value in kwargs.items():
            if isinstance(value, list):
                argv.append('--%s' % key)
                argv += value.split()
            elif isinstance(value, bool):
                # All boolean arguments in subcommands use action=store_true.
                argv.append('--%s' % key)
            else:
                argv.append('--%s' % key)
                argv.append(str(value))
        return argv

    def func(_, **kwargs):
        argv = kwargs_to_argv(kwargs)
        logging.info('Going to execute a command with command line: %s', argv)
        args = parser.parse_args(argv)
        # Additionally, assign values for global kwargs (such as rootdir).
        for key in global_kwargs:
            setattr(args, key, global_kwargs[key])
        subcommand(cursor, args)

    return func


class Dataframe:
    ''' 
    The Dataframe class incorporates all of the Shuffler's functionality. 
    It manages open databases in Shuffler schema and it has a method for every
    Shuffler subcommand.
    '''
    def __init__(self, in_db_path=None, rootdir='.'):
        '''
        Make a new dataframe by opening or creating a database with the 
        Shuffler schema.
        '''
        self._make_partial_subcommands()
        if in_db_path is None:
            self._create_new(rootdir=rootdir)
        else:
            self._load(in_db_path, rootdir=rootdir)

    def _make_partial_subcommands(self):
        '''
        Populate _partial_subcommands with Shuffler subcommand methods.
        Called only in __init__ and put into a separate function for clarity.
        '''
        parser = argparse.ArgumentParser()
        subcommands.add_subparsers(parser)

        self._partial_subcommands = []
        for subcommand in _collect_subcommands():
            func = partial(_make_subcommand_method,
                           subcommand=subcommand,
                           parser=parser)
            self._partial_subcommands.append((subcommand.__name__, func))

    def _update_subcommands(self):
        '''
        For every new / newly opened database, this method (re)creates a method
        for each Shuffler subcommand. 

        Methods are thin wrappers around Shuffler subcommands. 
        The wrappers hide "cursor" and "rootdir" from the user.
        Now user can just call "df.my_subcommand(wargs)" instead of
        "my_subcommand(cursor, rootdir, wargs)".

        Every time a dataframe is closed and (re)opened, wrappers need to be
        recreated for the new value of cursor and rootdir.
        '''
        for subcommand_name, func in self._partial_subcommands:
            setattr(Dataframe, subcommand_name,
                    func(cursor=self.cursor, rootdir=self.rootdir))

    def _create_new(self, rootdir):
        ''' Make a new empty database in a new file, and connect to it. '''
        self.temp_db_path = tempfile.NamedTemporaryFile().name
        self.rootdir = rootdir

        # Create an in-memory database.
        self.conn = sqlite3.connect(self.temp_db_path)
        backendDb.createDb(self.conn)
        self.cursor = self.conn.cursor()
        self._update_subcommands()

    def _load(self, in_db_path, rootdir):
        ''' Open an existing database that has the Shuffler schema. '''
        if in_db_path == ':memory:':
            self.temp_db_path = None
            self.conn = sqlite3.connect(':memory:')
            backendDb.createDb(self.conn)
        else:
            self.temp_db_path = tempfile.NamedTemporaryFile().name
            shutil.copyfile(in_db_path, self.temp_db_path)
            self.conn = sqlite3.connect(self.temp_db_path)

        self.rootdir = rootdir
        self.cursor = self.conn.cursor()
        self._update_subcommands()

    def _clean_up(self):
        ''' Close the connection to a database, and maybe delete tmp file.  '''
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.cursor = None
        if self.temp_db_path is not None:
            os.remove(self.temp_db_path)
        self.rootdir = None

    def __len__(self):
        ''' Return the number of images in the database. '''
        if self.conn is None:
            raise RuntimeError("Dataframe is empty.")
        self.cursor.execute("SELECT COUNT(1) FROM images")
        return self.cursor.fetchone()[0]

    def __getitem__(self, index):
        ''' 
        Get the i-th image entry with all its objects.
        Args:
          index:  the index of the image entry in the database.
        Returns:
          dictionary with the following keys.
          {
            'image':    Numpy array.
            'mask':     Numpy array.
            'imagefile' String.
            'maskfile'  String or None.
            'name'      String (e.g. when predicted by a classification model).
            'objects'   Dictionary with the same keys as field names
                        in table "objects" of the Shuffler schema.
          }
        '''
        if index >= len(self):
            raise IndexError("Index %d is greater than the dataframe size %d" %
                             (index, len(self)))
        self.cursor.execute("SELECT imagefile,maskfile,name FROM images")
        # TODO: figure out how to not load the whole database.
        image_entries = self.cursor.fetchall()
        imagefile, maskfile, name = image_entries[index]
        self.cursor.execute("SELECT * FROM objects WHERE imagefile=?",
                            (imagefile, ))
        objects = self.cursor.fetchall()
        imreader = backendMedia.MediaReader(rootdir=self.rootdir)
        if imagefile is not None:
            image = imreader.imread(imagefile)
        if maskfile is not None:
            mask = imreader.maskread(imagefile)
        return {
            'image': image,
            'mask': mask,
            'objects': [backendDb.objectAsDict(o) for o in objects],
            'imagefile': imagefile,
            'maskfile': maskfile,
            'name': name
        }

    def load(self, in_db_path, rootdir='.'):
        ''' Load a database in Shuffler format (after closing any open one.) '''
        self._clean_up()
        self._load(in_db_path, rootdir=rootdir)

    def save(self, out_db_path):
        ''' Save open database in Shuffler format, without closing. '''
        if self.temp_db_path is None:
            raise IOError("File was created in-memory. Can not save file.")
        self.conn.commit()
        shutil.move(self.temp_db_path, out_db_path)
        self.temp_db_path = None

    def close(self):
        ''' Close an open database. '''
        self._clean_up()
