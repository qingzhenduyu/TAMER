#!/bin/bash

version=$1
# 
for y in '2014' '2016' '2019' 
do
    echo '****************' start evaluating CROHME $y '****************'
    python eval/test.py data/crohme $version $y 320000 True
    echo 
done
