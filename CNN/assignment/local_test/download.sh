#!/bin/bash

#this downloads the zip file that contains the data
curl https://cs.brown.edu/courses/csci1470/hw_data/hw2.zip --output hw3.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip hw3.zip
# this cleans up the zip file, as we will no longer use it
rm hw3.zip

echo downloaded data
