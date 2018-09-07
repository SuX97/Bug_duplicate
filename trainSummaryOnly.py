from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import os
import time
import random
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Concatenate 
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

########## configurations ##########
BASE_PATH = "dataset"
DATA_PATH = "dataFiles"
RESULT_PATH = "results"
TOKENIZER_FILE_NAME = "tokenizer.pickle"
TRAIN_SET_FILE = "train.txt"
VAL_SET_FILE = "val.txt"
SUFFIX_SUMMARY = "_sum"
SUFFIX_DESC = "_desc"
WORD_EMBEDDING_PATH = "wordEmbedding"
WORD_EMBEDDING_FILE = "glove.6B.300d.txt"

MAX_SEQUENCE_LENGTH = 24 
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
LSTM_UNITS = 64

BASE_MODEL_BATCH_SIZE = 512
ALL_MODEL_BATCH_SIZE = 512
BASE_MODEL_EPOCHS = 2
ALL_MODEL_EPOCHS = 2
####################################

def buildWE():
    wordEmbeddingFile = os.path.join(os.path.join(BASE_PATH, WORD_EMBEDDING_PATH), WORD_EMBEDDING_FILE)
    embeddingsIndex = {}
    with open(wordEmbeddingFile) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = coefs
    return embeddingsIndex


def tripletLoss(features):
    margin = 1
    predA, predAP, predAN = features
    predA = K.l2_normalize(predA, axis=-1)
    predAP = K.l2_normalize(predAP, axis=-1)
    predAN = K.l2_normalize(predAN, axis=-1)
    posDistance = K.sum(predA * predAP, axis=-1)
    negDistance = K.sum(predA * predAN, axis=-1)
    loss = K.maximum(0.0, margin - posDistance + negDistance)
    return loss


def tripletOutputShape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def identicalLoss(predTrue, predY):
    return K.mean(predY)

def loadDataList(dataPath):
    dataList = []
    dataIndex = dict() 
    with open(dataPath, "r") as f:
        for line in f:
            bugA, bugAP, bugAN = line.rstrip().split()
            dataList.append([bugA, bugAP, bugAN])
            if bugA in dataIndex:
                dataIndex[bugA] += 1
            else:
                dataIndex[bugA] = 1

            if bugAP in dataIndex:
                dataIndex[bugAP] += 1
            else:
                dataIndex[bugAP] = 1

            if bugAN in dataIndex:
                dataIndex[bugAN] += 1
            else:
                dataIndex[bugAN] = 1
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

def genEmbeddingMatrix(wordIndex, embeddingsIndex):
    numWords = min(MAX_NUM_WORDS, len(wordIndex) + 1)
    embeddingMatrix = np.zeros((numWords, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddingsIndex.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embedding_vector
    return embeddingMatrix, numWords

def createBaseNetwork(numWords, embeddingMatrix):
    input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embeddedSequences = Embedding(numWords, EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False) (input)
    #embeddedSequences = Dropout(0.5)(embeddedSequences)
    #x = MaxPooling1D(5)(x)
    #x = Dropout(0.5)(x)
    #x = Conv1D(128, 1, activation='relu')(x)
    #x = GlobalMaxPooling1D()(x)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))(embeddedSequences)
    x = Dropout(0.5)(x)
    #x = Dense(256, activation='relu')(x)
    preds = Dense(128, activation='relu')(x)
    return Model(input, preds)

###################### Main #######################

ts = time.localtime(time.time())
tsStr = str(ts.tm_year)+str(ts.tm_mon)+str(ts.tm_mday)+str(ts.tm_hour)+str(ts.tm_min)+str(ts.tm_sec)
resultPath = os.path.join(RESULT_PATH, tsStr)
os.mkdir(resultPath)
os.system("cp *.py "+resultPath) ## save training scripts

########## data setup ############
trainDataList, trainDataIndex = loadDataList(os.path.join(BASE_PATH, TRAIN_SET_FILE))
valDataList, valDataIndex = loadDataList(os.path.join(BASE_PATH, VAL_SET_FILE))

trainDataContentSummary, trainDataContentDesc = loadData(trainDataIndex)
valDataContentSummary, valDataContentDesc = loadData(valDataIndex)

with open(os.path.join(BASE_PATH, TOKENIZER_FILE_NAME), 'rb') as handle:
    tokenizer = pickle.load(handle)

convertDataVector(trainDataContentSummary, trainDataContentDesc, tokenizer)
convertDataVector(valDataContentSummary, valDataContentDesc, tokenizer)

embeddingsIndex = buildWE()
embeddingMatrix, numWords = genEmbeddingMatrix(tokenizer.word_index, embeddingsIndex)

trainTriplets = []
for trainDataSlot in trainDataList:
    bugA, bugAP, bugAN = trainDataSlot
    trainTriplets.append([trainDataContentSummary[bugA], trainDataContentSummary[bugAP], trainDataContentSummary[bugAN]])

valTriplets = []
for valDataSlot in valDataList:
    bugA, bugAP, bugAN = valDataSlot
    valTriplets.append([valDataContentSummary[bugA], valDataContentSummary[bugAP], valDataContentSummary[bugAN]])
################################## 

########## create training Model ###########
inputA = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') 
inputAP = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
inputAN = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

baseNetwork = createBaseNetwork(numWords, embeddingMatrix) 
baseNetwork.summary()

featureA = baseNetwork(inputA)
featureAP = baseNetwork(inputAP)
featureAN = baseNetwork(inputAN)

distance = Lambda(tripletLoss, output_shape=tripletOutputShape)([featureA, featureAP, featureAN])

model = Model(inputs=[inputA, inputAP, inputAN], outputs=distance)
#############################################

###### prepare dataset for base model #######
trainA=[]
trainAP=[]
trainAN=[]
for item in trainTriplets:
    trainA.append(item[0].tolist())
    trainAP.append(item[1].tolist())
    trainAN.append(item[2].tolist())
valA=[]
valAP=[]
valAN=[]
for item in valTriplets:
    valA.append(item[0].tolist())
    valAP.append(item[1].tolist())
    valAN.append(item[2].tolist())
#############################################

###### train base network ########
baseFilePath = os.path.join(resultPath, "base-{epoch:02d}-{val_loss:.2f}.h5")
baseCP = ModelCheckpoint(baseFilePath, verbose=1, period=1)
baseCallbacks = [baseCP]
model.compile(loss=identicalLoss, optimizer=optimizers.RMSprop(lr=1e-4), metrics=None)
model.summary()
model.fit([trainA, trainAP, trainAN],
          np.ones(len(trainTriplets)).tolist(),
          batch_size=BASE_MODEL_BATCH_SIZE,
          epochs=BASE_MODEL_EPOCHS,
          validation_data=([valA, valAP, valAN], np.ones(len(valTriplets)).tolist()),
          callbacks=baseCallbacks)
##################################
baseNetworkPath = os.path.join(resultPath, "infoRetrieval.h5")
baseNetwork.save(baseNetworkPath)

###### prepare dataset for cls model ########
clsTrainA = []
clsTrainB = []
clsTrainLabel = []
for item in trainTriplets:
    clsTrainA.append(item[0].tolist())
    clsTrainB.append(item[1].tolist())
    clsTrainLabel.append(1)
    clsTrainA.append(item[0].tolist())
    clsTrainB.append(item[2].tolist())
    clsTrainLabel.append(0)

clsValA = []
clsValB = []
clsValLabel = []
for item in valTriplets:
    clsValA.append(item[0].tolist())
    clsValB.append(item[1].tolist())
    clsValLabel.append(1)
    clsValA.append(item[0].tolist())
    clsValB.append(item[2].tolist())
    clsValLabel.append(0)
#############################################

###### create classification model ##########
inputA = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
inputB = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

featureA = baseNetwork(inputA)
featureB = baseNetwork(inputB)

mergedFeature = Concatenate()([featureA, featureB])
mergedFeature = Dropout(0.3)(mergedFeature)
x = Dense(128, activation='relu')(mergedFeature)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
pred = Dense(1, activation='sigmoid')(x)

clsModel = Model(inputs=[inputA, inputB], outputs=pred)
#############################################

####### train classification network ########
allFilePath = os.path.join(resultPath, "all-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5")
allCP = ModelCheckpoint(allFilePath, verbose=1, period=1)
allCallbacks = [allCP]
clsModel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
clsModel.summary()
clsModel.fit([clsTrainA, clsTrainB],
             clsTrainLabel,
             batch_size=ALL_MODEL_BATCH_SIZE,
             epochs=ALL_MODEL_EPOCHS,
             validation_data=([clsValA, clsValB], clsValLabel),
             callbacks=allCallbacks)
#############################################
####################################################
