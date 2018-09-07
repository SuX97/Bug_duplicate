from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import os
import random

########## configurations ##########
BASE_PATH = "dataset"
PATH_BUG_STATISTICS = "bugStatistics"
PATH_BUG_TOTAL_INDEX_FILE = "bugIndex.txt"
PATH_BUG_DUP_LIST_FILE = "dupBugList.txt"
TRAIN_SET_FILE = "train.txt"
VAL_SET_FILE = "val.txt"
RANDOM_SAMPLE_NUM_FROM_TOTAL = 100
####################################

###################### Main #######################

### Load bug index
bugIndex = []
with open(os.path.join(os.path.join(BASE_PATH, PATH_BUG_STATISTICS), PATH_BUG_TOTAL_INDEX_FILE), "r") as f:
    for line in f:
        bugIndex.append(line.rstrip())

### Load duplicated bug list
dupBugList = []
with open(os.path.join(os.path.join(BASE_PATH, PATH_BUG_STATISTICS), PATH_BUG_DUP_LIST_FILE), "r") as f:
    for line in f:
        dupLine = []
        for item in line.rstrip().split():
            dupLine.append(item)
        dupBugList.append(dupLine)

for dupliet in dupBugList:
    for item in dupliet:
        if not (item in bugIndex):
                print (item)

####################################################
