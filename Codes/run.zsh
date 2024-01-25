#!/bin/zsh

N=0
for folder in /data/Images_IMAJ/Fonds*
do
    for subfolder in "$folder"/*
    do  
        numactl -C $(($N % 64)) python /data/Images_IMAJ/Color_features_extraction.py $subfolder $subfolder.csv &
        ((N++))
    done
done
