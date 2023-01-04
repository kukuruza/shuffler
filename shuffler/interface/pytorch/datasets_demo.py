'''
A demo for running PyTorch inference on a database, and recording some results.
'''

import os
import shutil
import tempfile
import torch
import torchvision.transforms

from shuffler.utils import testing as testing_utils
from shuffler.interface.pytorch import datasets


class ConvNet(torch.nn.Module):
    input_size = (28, 28)

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


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
        torchvision.transforms.Resize(ConvNet.input_size),
    ])
    # A transform for name: make it categorical.
    transform_name = lambda x: 1 if x == 'bus' else 0

    # Make a dataset of OBJECTS. Every returned item is an object in the db.
    # We specify mode='w' because we want to record some values.
    dataset = datasets.ObjectDataset(
        tmp_in_db_file,
        rootdir=rootdir,
        mode='w',
        used_keys=['image', 'objectid', 'name'],
        transform_group={
            'image': transform_image,
            'name': transform_name,
            # 'objectid' does not need a transform, its type "int" suits us.
        })

    batch_size = 2  # All 3 samples are broken down into 2 batches.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1)

    model = ConvNet(num_classes=2)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #### Training (for 1 epoch). ####

    num_epochs = 1
    for _ in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            logps = model.forward(batch['image'])
            loss = criterion(logps, batch['name'])
            loss.backward()
            optimizer.step()

    #### Inference (on the same data). ####

    for batch in data_loader:
        images = batch['image']
        objectids = batch['objectid']
        names = batch['name']

        # Inference on a batch of images.
        results = model(images)

        # Convert from torch.Tensor to numpy arrays.
        results = results.data.numpy().argmax(axis=1)
        objectids = objectids.numpy()
        names = names.numpy()

        # Sample-by-sample inside a batch.
        for objectid, name, result in zip(objectids, names, results):

            print('%s with objectid=%d produced dummy result %f.' %
                  (name, objectid, result))

            # Write the result to the database (if desired).
            dataset.addRecord(objectid, 'result', str(result))

    # Close the dataset to save changes.
    dataset.close()

    # Clean up.
    os.remove(tmp_in_db_file)


if __name__ == '__main__':
    main()
