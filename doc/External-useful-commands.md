## Working wth `ffmepg`.

```bash
# Re-encode images using the current default parameters (jpeg-like, good for natural images).
ffmpeg -i testdata/cars/images.avi -c:v mjpeg -pix_fmt yuvj444p /tmp/images.avi

# Re-encode masks using the current default parameters (lossless, good for png).
ffmpeg -i testdata/cars/masks.avi -c:v huffyuv -pix_fmt rgb24 /tmp/masks.avi

# Read in ffmpeg without output.
ffmpeg -i testdata/cars/masks.avi  -f null /dev/null
```

Imagemagick useful commands
```bash
# Make a 3xN grid out of images, with 2 pixels on each side:
montage -density 300 -tile 3x0 -geometry +2+2 -border 0 -background none snapshots/*.png snapshots.png
```
