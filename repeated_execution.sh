#!/bin/bash
: '
This script executes iteratively n times the HOG algorithm on the images
contained in ./images directory, where n is a value specified by the user.
'
declare -i rounds=$1


for round in $(seq 1 $rounds);
do
    for img in $(seq 1 12);
    do  
        if [ $img -le 10 ]
        then
            img_num="0$img";
        else
            img_num="$img";
        fi
        ./main -i "images/crop0010${img_num}.png"
    done
done