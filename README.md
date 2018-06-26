# shuffler
Toolbox for manipulating image annotations in computer vision

# Other useful commands.
```
picsdir=4096x2160_5Hz__2018_6_15_10_10_38
ffmpeg -f image2 -r 5  -pattern_type glob -i ${picsdir}'/*.jpg' -r 5 -c:v mpeg4 -vtag xvid ${picsdir}.avi
```

# Import labelled images and display them with mask and frame_id.
```
./pipeline.py \
  importPictures --image_pattern "path/to/imagedir/*.jpg" --mask_pattern "path/to/labeldir/*.png" \
  display --labelmap_file /path/to/labelmap.json --winwidth 1500
```

# Import labelled images and write them with mask and frame_id.
```
./pipeline.py \
  importPictures --image_pattern "path/to/imagedir/*.jpg" --mask_pattern "path/to/labeldir/*.png" \
  writeVideo ~/datasets/scotty/4096x2160_5Hz__2018_6_7_18_57_38_masked.avi --labelmap_file /path/to/labelmap.json --fps 1
```
