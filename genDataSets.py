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
TEST_SET_FILE = "test.txt"
VAL_SPLIT = 0.01
TEST_SPLIT = 0.01
RANDOM_SAMPLE_NUM_FROM_TOTAL = 10
####################################

def genDataSetFile(targetFileName, targetList, targetIndex):
    targetFile = open(os.path.join(BASE_PATH, targetFileName), "w")

    indexL1 = 1.0
    for dupliet in targetList:
        base = dupliet[0]
        for pos in dupliet[1:]:
            sampledBugIndex = random.sample(targetIndex, RANDOM_SAMPLE_NUM_FROM_TOTAL)
            for neg in sampledBugIndex:
                if not (neg in dupliet):
                    ### add a sample with [A, A_POSITIVE, A_NEGTIVE]
                    targetFile.write(base+' '+pos+' '+neg+'\n')
                    print ("[%5.1f%%]" % (indexL1/len(targetList)*100), base, pos, neg)
        indexL1 += 1
    targetFile.close()

def genBugIndex(targetBugList):
    targetIndex = []
    for dupliet in targetBugList:
        for item in dupliet:
            if not (item in targetIndex):
                targetIndex.append(item)
    return targetIndex

###################### Main #######################

### Load bug index ###
print ("[Info]: Load bug index file...")
bugIndex = []
with open(os.path.join(os.path.join(BASE_PATH, PATH_BUG_STATISTICS), PATH_BUG_TOTAL_INDEX_FILE), "r") as f:
    for line in f:
        bugIndex.append(line.rstrip())
######################

### Load duplicated bug list ###
print ("[Info]: Load duplicated bug group file...")
dupBugList = []
with open(os.path.join(os.path.join(BASE_PATH, PATH_BUG_STATISTICS), PATH_BUG_DUP_LIST_FILE), "r") as f:
    for line in f:
        dupLine = []
        for item in line.rstrip().split():
            dupLine.append(item)
        dupBugList.append(dupLine)
################################

### Split train, val, test #####
print ("[Info]: Split train/val/test set...")
valDupBugLength = int(len(dupBugList) * VAL_SPLIT)
testDupBugLength = int(len(dupBugList) * TEST_SPLIT)
trainDupBugLength = len(dupBugList) - valDupBugLength - testDupBugLength

random.shuffle(dupBugList)

trainDupBugList = dupBugList[0:trainDupBugLength]
valDupBugList = dupBugList[trainDupBugLength:trainDupBugLength+valDupBugLength]
testDupBugList = dupBugList[trainDupBugLength+valDupBugLength:] 

trainDupBugIndex = genBugIndex(trainDupBugList)
valDupBugIndex = genBugIndex(valDupBugList)
testDupBugIndex = genBugIndex(testDupBugList)

valTestDupBugIndex = valDupBugIndex + testDupBugIndex
trainBugIndex = []
for item in bugIndex:
    if not (item in valTestDupBugIndex):
        trainBugIndex.append(item)

testTrainDupBugIndex = trainDupBugIndex + testDupBugIndex
valBugIndex = []
for item in bugIndex:
    if not (item in testTrainDupBugIndex):
        valBugIndex.append(item)

valTrainDupBugIndex = trainDupBugIndex + valDupBugIndex
testBugIndex = []
for item in bugIndex:
    if not (item in valTrainDupBugIndex):
        testBugIndex.append(item)
################################
print ("[Info]: Generate final dataset files...")
genDataSetFile(TRAIN_SET_FILE, trainDupBugList, trainBugIndex)
genDataSetFile(VAL_SET_FILE, valDupBugList, valBugIndex)
genDataSetFile(TEST_SET_FILE, testDupBugList, testBugIndex)

####################################################
