Import from LabelMe, each image is labelled by multiple annotators.

```bash
python -m shuffler --rootdir '.' -i 'testdata/labelme/init_v5.db' \
  importLabelmeObjects --annotations_dir 'testdata/labelme/w55-e04-objects1' \
  --keep_original_object_name --polygon_name annotator1 \| \
  importLabelmeObjects --annotations_dir 'testdata/labelme/w55-e04-objects2' \
  --keep_original_object_name --polygon_name annotator2 \| \
  importLabelmeObjects --annotations_dir 'testdata/labelme/w55-e04-objects3' \
  --keep_original_object_name --polygon_name annotator3 \| \
  mergeIntersectingObjects \| \
  polygonsToMask --mask_pictures_dir 'testdata/labelme/mask_polygons' --skip_empty_masks \| \
  dumpDb --tables objects polygons \| \
  examineImages --mask_aside
```
