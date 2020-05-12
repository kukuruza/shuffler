import os
import tempfile
from dataframe import Dataframe
import matplotlib.pyplot as plt

df = Dataframe()
df.load('testdata/cars/micro1_v4.db', rootdir='testdata/cars')
df.sql(sql='DELETE FROM properties')
tmp_file_path = tempfile.NamedTemporaryFile().name
df.save(tmp_file_path)
df.printInfo()
df.load('testdata/cars/micro1_v4.db', rootdir='testdata/cars')
df.printInfo()
df.load(tmp_file_path, rootdir='testdata/cars')
df.printInfo()
df.displayImagesPlt(limit=4, with_objects=True, with_imagefile=True)
plt.show()
print(len(df))
image, objects = df[1]['image'], df[1]['objects']
plt.imshow(image)
plt.show()
df.close()

os.remove(tmp_file_path)