#!/bin/sh

# this will ask for the path of the dataset directory
echo "Specify the Dataset Directory Path"
read -p "PATH:" path

# can improve by making it can pass an input into it
python3 train.py -data_path path