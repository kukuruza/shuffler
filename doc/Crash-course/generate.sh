#!/bin/bash

convert doc/Crash-course/cars-micro-image1.png -resize x500 \
    doc/Crash-course/cars-micro-image1-x500.png

convert -size 1500x500 xc:white \
    doc/Crash-course/cars-micro-image1-x500.png -gravity center -composite \
    doc/Crash-course/cars-micro-image1-1500x500.png