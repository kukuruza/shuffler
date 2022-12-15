'''
This is an interface between Shuffler operations and ipython notebook.

It introduces Dataframe class, that is the only Shuffler interface class
for an IPython notebook. Every Shuffler operation is a method in the Dataframe
class.

The workflow is expected to be like below. Refer to "dataframe_demo.ipynb" file.

# Init.
df = Dataframe()
df.load('testdata/cars/micro1_v5.db', rootdir='testdata/cars')

# Shuffler operations.
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

from shuffler import operations
from shuffler.backend import backend_db
from shuffler.backend import backend_media


def _collect_operations():
    '''
    Collect all operation functions and their parsers from operation modules.
    '''
    # Iterate modules (files) in operations package.
    all_functions = []
    for module_name in dir(operations):
        module = getattr(operations, module_name)

        functions = inspect.getmembers(module)
        functions = [f for f in functions if inspect.isfunction(f[1])]
        functions_dict = dict(functions)

        # Iterate functions in a module.
        for subparser_name in functions_dict:
            if not subparser_name.endswith('Parser'):
                continue
            operation_name = subparser_name[:-len('Parser')]
            if not operation_name in functions_dict:
                logging.error('Weird: have function %s, but not function %s',
                              subparser_name, operation_name)
                continue
            getattr(operations, module_name)

            all_functions.append(functions_dict[operation_name])

    return all_functions


def _make_operation_method(cursor, operation, parser, **global_kwargs):
    '''
    Given a operation method and its parser method, add a class method that
    would parse the input **kwargs and call the operation.

    This whole function is a struggle with the argparse package. In particular,
    it is a interface between **kwargs arguments that the DataFrame takes and
    the parsed arguments that operations take.
    '''

    def kwargs_to_argv(kwargs):
        ''' Get arguments as a dictionary, and make an input for the parser. '''
        argv = [operation.__name__]
        for key, value in kwargs.items():
            if isinstance(value, list):
                argv.append('--%s' % key)
                argv += value.split()
            elif isinstance(value, bool):
                # All boolean arguments in operations use action=store_true.
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
        operation(cursor, args)

    return func


class Dataframe:
    '''
    The Dataframe class incorporates all of the Shuffler's functionality.
    It manages open databases in Shuffler schema and it has a method for every
    Shuffler operation.
    '''

    def __init__(self, in_db_path=None, rootdir='.'):
        '''
        Make a new dataframe by opening or creating a database with the
        Shuffler schema.
        '''
        self._make_partial_operations()
        if in_db_path is None:
            self._create_new(rootdir=rootdir)
        else:
            self._load(in_db_path, rootdir=rootdir)

    def _make_partial_operations(self):
        '''
        Populate _partial_operations with Shuffler operation methods.
        Called only in __init__ and put into a separate function for clarity.
        '''
        parser = argparse.ArgumentParser()
        operations.add_subparsers(parser)

        self._partial_operations = []
        for operation in _collect_operations():
            func = partial(_make_operation_method,
                           operation=operation,
                           parser=parser)
            self._partial_operations.append((operation.__name__, func))

    def _update_operations(self):
        '''
        For every new / newly opened database, this method (re)creates a method
        for each Shuffler operation.

        Methods are thin wrappers around Shuffler operations.
        The wrappers hide "cursor" and "rootdir" from the user.
        Now user can just call "df.my_operation(wargs)" instead of
        "my_operation(cursor, rootdir, wargs)".

        Every time a dataframe is closed and (re)opened, wrappers need to be
        recreated for the new value of cursor and rootdir.
        '''
        for operation_name, func in self._partial_operations:
            setattr(Dataframe, operation_name,
                    func(cursor=self.cursor, rootdir=self.rootdir))

    def _create_new(self, rootdir):
        ''' Make a new empty database in a new file, and connect to it. '''
        self.temp_db_path = tempfile.NamedTemporaryFile().name
        self.rootdir = rootdir

        # Create an in-memory database.
        self.conn = sqlite3.connect(self.temp_db_path)
        backend_db.createDb(self.conn)
        self.cursor = self.conn.cursor()
        self._update_operations()

    def _load(self, in_db_path, rootdir):
        ''' Open an existing database that has the Shuffler schema. '''
        if in_db_path == ':memory:':
            self.temp_db_path = None
            self.conn = sqlite3.connect(':memory:')
            backend_db.createDb(self.conn)
        else:
            self.temp_db_path = tempfile.NamedTemporaryFile().name
            shutil.copyfile(in_db_path, self.temp_db_path)
            self.conn = sqlite3.connect(self.temp_db_path)

        self.rootdir = rootdir
        self.cursor = self.conn.cursor()
        self._update_operations()

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
        cols = backend_db.getColumnsInTable(self.cursor, 'objects')
        self.cursor.execute(
            "SELECT %s FROM objects WHERE imagefile=?" % ','.join(cols),
            (imagefile, ))
        objects = self.cursor.fetchall()
        imreader = backend_media.MediaReader(rootdir=self.rootdir)
        if imagefile is not None:
            image = imreader.imread(imagefile)
        if maskfile is not None:
            mask = imreader.maskread(imagefile)
        return {
            'image': image,
            'mask': mask,
            'objects': [zip(cols, o) for o in objects],
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
