#!/bin/bash

# Delete all files within the 'inputs' directory and its subdirectories
find inputs/ -type f -delete

# Delete all files within the 'outputs' directory and its subdirectories
find outputs/ -type f -delete

echo "Cleaned 'inputs' and 'outputs' directories (files only)."