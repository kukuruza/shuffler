import os
import tempfile
from dataframe import Dataframe

df = Dataframe(rootdir='testdata/cars')
df.load('testdata/cars/micro1_v4.db')
df.sql(sql='DELETE FROM properties')
tmp_file_path = tempfile.NamedTemporaryFile()
df.save(tmp_file_path)
df.printInfo()
df.load('testdata/cars/micro1_v4.db')
df.printInfo()
df.load(tmp_file_path)
df.printInfo()
df.displayImagesPlt(limit=4, with_objects=True, with_imagefile=True)
df.close()

os.remove(tmp_file_path)