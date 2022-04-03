import os
import tempfile
from interface.dataframe import Dataframe
import matplotlib.pyplot as plt

df = Dataframe()
df.load('testdata/cars/micro1_v5.db', rootdir='testdata/cars')
df.sql(sql='DELETE FROM properties')
tmp_file_path = tempfile.NamedTemporaryFile().name
df.save(tmp_file_path)
df.printInfo()
df.load('testdata/cars/micro1_v5.db', rootdir='testdata/cars')
df.printInfo()
df.load(tmp_file_path, rootdir='testdata/cars')
df.printInfo()
# Create a figure before calling imaging function in order to use Figure object.
fig = plt.figure(figsize=(5, 15))
df.displayImagesPlt(limit=4, with_objects=True, with_imagefile=True)
plt.show()

# Create a figure before calling imaging function in order to use Figure object.
fig = plt.figure(figsize=(15, 5))
df.plotHistogram(sql='SELECT width FROM objects')
plt.show()

print(len(df))
image, objects = df[1]['image'], df[1]['objects']
plt.imshow(image)
plt.show()
df.close()

os.remove(tmp_file_path)