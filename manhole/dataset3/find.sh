#!/bin/bash

# 比较两个文件夹中的文件，并将不同的文件复制到指定文件夹

folder1="/home/stoair/data/dataset3/img"
folder2="/home/stoair/data/dataset3/add"
output_folder="./different"

# 比较两个文件夹中的文件，并将不同的文件复制到指定文件夹
diff_files=$(diff <(cd $folder1 && find . -type f | sort) <(cd $folder2 && find . -type f | sort) | grep "^>" | sed 's/^> //')

for file in $diff_files; do
    cp $folder1/$file $output_folder
    cp $folder2/$file $output_folder
done

