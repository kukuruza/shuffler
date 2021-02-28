'''
A demo for running inference on a database, and recording some results.
'''

import os
import shutil
import tempfile
import torch
import torchvision.transforms

from lib.utils import testUtils
from interface.pytorch import datasets


def dummyPredict(batch):
    ''' 
    Replace this function with a real inference logic in your code. 

    Args:
        batch:  (torch.Tensor of size (batch_size, 3, Y, X)) A batch of images.
    Returns:
        torch.Tensor of size (batch_size, )
    '''
    return torch.rand(batch.size()[0])


def main():
    # This database contains 3 images with 2 cars and 1 bus.
    in_db_file = testUtils.Test_carsDb.CARS_DB_PATH
    rootdir = testUtils.Test_carsDb.CARS_DB_ROOTDIR

    # We are going to make changes to the database, so let's work on its copy.
    tmp_in_db_file = tempfile.NamedTemporaryFile().name
    shutil.copy(in_db_file, tmp_in_db_file)

    # A transform, as is normally used in Pytorch.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(80, scale=(0.8, 1.0)),
    ])

    # Make a dataset of OBJECTS. Every returned item is an object in the db.
    # We specify mode='w' because we want to record some values.
    dataset = datasets.ObjectDataset(tmp_in_db_file,
                                     rootdir=rootdir,
                                     mode='w',
                                     used_keys=['image', 'objectid', 'name'],
                                     transform_group={'image': transform})

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=1)

    for batch in data_loader:

        images = batch['image']
        objectids = batch['objectid']
        names = batch['name']

        # Replace this with the real inference logic in your code.
        results = dummyPredict(images)

        # At this point, "images", "objectids", "names", "results" are Tensors.

        for objectid, name, result in zip(objectids, names, results):

            print('%s with objectid=%d produced dummy result %f.' %
                  (name, objectid, result))

            # Get a value out of a Tensor with a single element.
            objectid = objectid.item()
            # Write the result to the database (if desired).
            dataset.addRecord(objectid, 'result', str(result))

    # Close the dataset to save changes.
    dataset.close()

    # Clean up.
    os.remove(tmp_in_db_file)


if __name__ == '__main__':
    main()
