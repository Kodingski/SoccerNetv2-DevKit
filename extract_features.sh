#!/bin/sh

search_dir=Data/matches/
for entry in "$search_dir"/*
do
  python Features/VideoFeatureExtractor.py --path_video "$entry" --path_features "$entry.npy" --overwrite --FPS 10

done