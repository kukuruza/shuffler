# shuffler
Toolbox for manipulating image annotations in computer vision

# Other useful commands.
```
picsdir=4096x2160_5Hz__2018_6_15_10_10_38
ffmpeg -f image2 -r 5  -pattern_type glob -i ${picsdir}'/*.jpg' -r 5 -c:v mpeg4 -vtag xvid ${picsdir}.avi
```
