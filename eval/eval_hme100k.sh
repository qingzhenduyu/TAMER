#!/bin/bash
version=$1
# 
python eval/test.py data/hme100k $version test 480000 False

