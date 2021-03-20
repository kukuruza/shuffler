'''
A demo for running Keras inference on a database, and recording some results.
'''

import os
import shutil
import tempfile
import numpy as np

from lib.utils import testUtils
from interface.keras import generators


def dummyPredict(batch):
    ''' 
    Replace this function with a real inference logic in your code. 

    Args:
        batch:  (a list of numpy arrays of size (Y, X, 3)) A batch of images.
    Returns:
        numpy array of size (batch_size, )
    '''
    return np.random.rand(len(batch))


def main():
    # This database contains 3 images with 2 cars and 1 bus.
    in_db_file = testUtils.Test_carsDb.CARS_DB_PATH
    rootdir = testUtils.Test_carsDb.CARS_DB_ROOTDIR

    # We are going to make changes to the database, so let's work on its copy.
    tmp_in_db_file = tempfile.NamedTemporaryFile().name
    shutil.copy(in_db_file, tmp_in_db_file)

    # Make a generator of OBJECTS. Every returned item is an object in the db.
    # We specify mode='w' because we want to record some values.
    generator = generators.ObjectGenerator(
        tmp_in_db_file,
        rootdir=rootdir,
        mode='w',
        used_keys=['image', 'objectid', 'name'],
        batch_size=2,
        shuffle=False)

    for batch in generator:

        images = batch[0]
        objectids = batch[1]
        names = batch[2]

        # Replace this with the real inference logic in your code.
        results = dummyPredict(images)

        # At this point, "images", "objectids", "names", "results" are.

        for objectid, name, result in zip(objectids, names, results):

            print('%s with objectid=%d produced dummy result %f.' %
                  (name, objectid, result))

            # Write the result to the database (if desired).
            generator.addRecord(objectid, 'result', str(result))

    # Close the generator to save changes.
    generator.close()

    # Clean up.
    os.remove(tmp_in_db_file)


if __name__ == '__main__':
    main()
