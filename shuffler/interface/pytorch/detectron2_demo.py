'''
A demo for creating a Detectron2 dataset based on Shuffler db.
'''

import shutil
import tempfile
import torchvision.transforms
import detectron2

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import detectron2 as shuffler_detectron2


def main():
    # This database contains 3 images with 2 cars and 1 bus.
    in_db_file = testing_utils.Test_carsDb.CARS_DB_PATH
    rootdir = testing_utils.Test_carsDb.CARS_DB_ROOTDIR

    # We are going to make changes to the database, so let's work on its copy.
    tmp_in_db_file = tempfile.NamedTemporaryFile().name
    shutil.copy(in_db_file, tmp_in_db_file)

    # A transform, as is normally used in Pytorch.
    transform_image = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
    ])
    # A transform for name: make it categorical.
    transform_name = lambda x: 1 if x == 'bus' else 0

    # Register the dataset. Need to register a dataset for each db file.
    shuffler_detectron2.register_object_dataset(
        "shuffler_object_dataset",
        tmp_in_db_file,
        rootdir=rootdir,
        used_keys=['image', 'objectid', 'name'],
        transform_group={
            'image': transform_image,
            'name': transform_name,
            # 'objectid' does not need a transform, its type "int" suits us.
        })
    # Get the registered dataset.
    dataset = detectron2.data.DatasetCatalog.get("shuffler_object_dataset")

    print('Dataset has %d objects.' % len(dataset))
    sample = dataset[0]  # The first object.
    print('The first image has dimensions: %s.' % str(sample['image'].size()))
    print('The first object id is: %d.' % sample['objectid'])
    print('The first name category is: %d. (0: car, 1: bus.)' % sample['name'])


if __name__ == '__main__':
    main()
