#!/bin/sh

# this is making an inference using a bash script
python3 inference.py
echo "Specify a path of sample image for ConvNextV2 inference"
read -p "PATH: " img_path

python3 inference.py -img_path img_path