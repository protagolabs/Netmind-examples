#!/bin/bash

mkdir bean_data
cd bean_data

wget https://storage.googleapis.com/ibeans/train.zip
unzip train.zip
rm train.zip

wget https://storage.googleapis.com/ibeans/validation.zip
unzip validation.zip
rm validation.zip

wget https://storage.googleapis.com/ibeans/test.zip
unzip test.zip
rm test.zip

cd ..