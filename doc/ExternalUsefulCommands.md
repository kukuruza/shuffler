## Working wth `ffmepg`.

```
# Re-encode images using the current default parameters (jpeg-like, good for natural images).
ffmpeg -i test/cars/images.avi -c:v mjpeg -pix_fmt yuvj444p /tmp/images.avi

# Re-encode masks using the current default parameters (lossless, good for png).
ffmpeg -i test/cars/masks.avi -c:v huffyuv -pix_fmt rgb24 /tmp/masks.avi

# Read in ffmpeg without output.
ffmpeg -i test/cars/masks.avi  -f null /dev/null
```
