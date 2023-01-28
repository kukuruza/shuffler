An unordered list of TODO items.

- ImageNet2012 io.
- Cover all operations with unit tests.
- "Filter images" operations should either remove images with objects or create new tables with filtered images with objects, and retire old tables, based on what is going to take less time.
- should not load a database to memory for large databases, instead it should copy the database to a temp location.
- Rework keyboard config.
  - Add user config to ~/.shufflerrc.
  - Add default config to "toml".
  - Add default config file to MANIFEST.
  - Use "importlib.resources" ro read the config file.
  - Use rich for beatiuful terminal output.
- Add matplotlib implementation of seaborn plots to eliminate seaborn dependencies:
  - Violin: https://www.marsja.se/how-to-make-a-violin-plot-in-python-using-matplotlib-and-seaborn/
  - Countplot: https://stackoverflow.com/questions/55667185/what-is-matplotlibs-alternative-for-countplot-from-seaborn
  - Stripplot is not replaceable with seaborn.
- Configure colormap for plotting boxes.
- 
