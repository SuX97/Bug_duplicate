from __future__ import absolute_import
from __future__ import print_function

import os
import random
import pickle
from keras.preprocessing.text import Tokenizer

########## configurations ##########
BASE_PATH = "dataset"
DATA_PATH = "dataFiles"
PATH_BUG_STATISTICS = "bugStatistics"
PATH_BUG_TOTAL_INDEX_FILE = "bugIndex.txt"

SUFFIX_SUMMARY = "_sum"
SUFFIX_DESC = "_desc"

MAX_NUM_WORDS = 20000
####################################

def loadData(dataIndex):
    dataContentSummary = dict()
    dataContentDesc = dict()
    for dataFileName in dataIndex:
        with open(os.path.join(os.path.join(BASE_PATH, DATA_PATH), dataFileName + SUFFIX_SUMMARY), "r") as f:
            dataContentSummary[dataFileName] = f.read().rstrip()
        with open(os.path.join(os.path.join(BASE_PATH, DATA_PATH), dataFileName + SUFFIX_DESC), "r") as f:
            dataContentDesc[dataFileName] = f.read().rstrip()
    return dataContentSummary, dataContentDesc

def genTokenizer(fitData):
    texts = []
    for data in fitData:
        texts += [v for v in data.values()]
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    return tokenizer
    
###################### Main #######################
bugIndex = []
with open(os.path.join(os.path.join(BASE_PATH, PATH_BUG_STATISTICS), PATH_BUG_TOTAL_INDEX_FILE), "r") as f:
    for line in f:
        bugIndex.append(line.rstrip())

bugContentSummary, bugContentDesc = loadData(bugIndex)

tokenizer = genTokenizer([bugContentSummary, bugContentDesc])

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
####################################################
