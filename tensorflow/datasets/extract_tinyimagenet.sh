#!/bin/bash
#

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

rm tiny-imagenet-200.zip

python tinyimagenet_val_format.py

