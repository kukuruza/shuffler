# The 1st argument is the directory with directories of images.
# The second argument is the divider objectid for background colors.

# Will make a directory with image collages.

cd $1
for folder in ./*; do

  if [[ -d $folder ]]; then

    cd ${folder}
    for filename in ?????????.jpg; do
      basename=$(echo ${filename} | cut -c1-9)
      echo ${basename}
      if [ ${basename} -le $2 ]; then
          undercolor="PaleTurquoise"
      else
          undercolor="white"
      fi
      convert "${filename}" \
              -resize "500x500" -gravity center -extent 500x500 \
              -gravity north -extent 500x530 \
              -pointsize 20 -fill black -undercolor ${undercolor} -gravity south -annotate +0+0 "${basename}" \
              "${basename}_id.jpg"
    done
    cd ..

    montage ${folder}/*_id.jpg -geometry 500x550+20+20 "${folder}.jpg"

    rm ${folder}/*_id.jpg

  fi
done

cd ..
mkdir -p $1-collage
rm -f $1-collage/*
mv $1/*.jpg $1-collage
cd ..
