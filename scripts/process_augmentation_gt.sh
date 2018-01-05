#!/usr/bin/env bash

python ././../examples/hed/data_preprocessing.py data_augmentation --input-image-path ././../data/new_dataset/crop/
python ././../examples/hed/data_preprocessing.py data_augmentation --input-image-path ././../data/new_dataset/orig/
python ././../examples/hed/data_preprocessing.py get_ground_truth --input-image-path ././../data/new_dataset/crop/augmentation/
