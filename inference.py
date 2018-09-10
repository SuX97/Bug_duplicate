from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import os
import random
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
########## configurations ##########
BASE_PATH = "dataset"
DATA_PATH = "dataFiles"
MODEL_PATH = "models"
MODEL_FILE_NAME = "all-01-0.44-0.85.h5"
TOKENIZER_FILE_NAME = "tokenizer.pickle"
TEST_SET_FILE = "inference.txt"
SUFFIX_SUMMARY = "_sum"
SUFFIX_DESC = "_desc"

FP_THRESHOLD = 0.85

MAX_SEQUENCE_LENGTH = 24 
####################################

def loadDataList(dataPath):
    dataList = []
    dataIndex = dict() 
    with open(dataPath, "r") as f:
        for line in f:
            bugA = line.rstrip().split()
            dataList.append([bugA, bugAP, bugAN])
            if bug in dataIndex:
                dataIndex[bug] += 1
			
    return dataList, dataIndex


def loadData(dataIndex):
    dataContentSummary = dict()
    dataContentDesc = dict()
    for dataFileName in dataIndex:
        with open(os.path.join(os.path.join(BASE_PATH, DATA_PATH), dataFileName + SUFFIX_SUMMARY), "r") as f:
            dataContentSummary[dataFileName] = f.read().rstrip()
        with open(os.path.join(os.path.join(BASE_PATH, DATA_PATH), dataFileName + SUFFIX_DESC), "r") as f:
            dataContentDesc[dataFileName] = f.read().rstrip()
    return dataContentSummary, dataContentDesc

def convertDataVector(dataContentSummary, dataContentDesc, tokenizer):
    tempBuf = []
    for key, value in dataContentSummary.items():
        tempBuf.append(value)
    for key, value in dataContentDesc.items():
        tempBuf.append(value)
    # add a fake item with MAX_SEQUENCE_LENGTH to ensure all sequence will be as this size
    tempBuf.append([0]*MAX_SEQUENCE_LENGTH) 
    seqs = tokenizer.texts_to_sequences(tempBuf)
    seqs = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
    i = 0
    for key, value in dataContentSummary.items():
        dataContentSummary[key] = seqs[i]
        i = i+1
    for key, value in dataContentDesc.items():
        dataContentDesc[key] = seqs[i]
        i = i+1

###################### Main #######################

########## data setup ############
testDataList, testDataIndex = loadDataList(os.path.join(BASE_PATH, TEST_SET_FILE))

testDataContentSummary, testDataContentDesc = loadData(testDataIndex)

with open(os.path.join(BASE_PATH, TOKENIZER_FILE_NAME), 'rb') as handle:
    tokenizer = pickle.load(handle)

convertDataVector(testDataContentSummary, testDataContentDesc, tokenizer)

testPair = []
for testDataA in testDataList:
	for testDataB in testDataList:
		bugA = testDataA
		bugB = testDataB
		testPair.append([testDataContentSummary[bugA],testDataContentSummary[bugB]]) 
'''
for testDataSlot in testDataList:
    bugA, bugAP, bugAN = testDataSlot
    testTriplets.append([testDataContentSummary[bugA], testDataContentSummary[bugAP], testDataContentSummary[bugAN]])
################################## 
'''
###### prepare dataset for cls model ########
clsTestA = []
clsTestB = []
#clsTestLabel = []
for item in testTriplets:
    clsTestA.append(item[0].tolist())
    clsTestB.append(item[1].tolist())
#    clsTestLabel.append(1)
    clsTestA.append(item[0].tolist())
    clsTestB.append(item[2].tolist())
#    clsTestLabel.append(0)
#############################################

######## load pre-trained cls Model #########
modelFileName = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
clsModel = load_model(modelFileName)
#############################################

############## run predict ###############
predResult = clsModel.predict(x=[clsTestA, clsTestB], batch_size=1024, verbose=1)
print(predResult)
#############################################
####################################################
