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
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Concatenate, GlobalMaxPooling1D 
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

INPUT_MODE = "SUM"   # Currently support two modes: 1.SUM 2.SUM+DESC
MAX_BUG_SUMMARY_SEQUENCE_LENGTH = 24
MAX_BUG_DESC_SEQUENCE_LENGTH = 512 
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
LSTM_UNITS = 256

BASE_MODEL_BATCH_SIZE = 512
ALL_MODEL_BATCH_SIZE = 512
BASE_MODEL_EPOCHS = 2
ALL_MODEL_EPOCHS = 2
####################################

############ Functions #############
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
        if INPUT_MODE == 'SUM+DESC':
            with open(os.path.join(os.path.join(BASE_PATH, DATA_PATH), dataFileName + SUFFIX_DESC), "r") as f:
                dataContentDesc[dataFileName] = f.read().rstrip()
    return dataContentSummary, dataContentDesc

def convertDataVector(dataContent, seq_length, tokenizer):
    tempBuf = []
    for key, value in dataContent.items():
        tempBuf.append(value)
    # add a fake item with seq_length to ensure all sequence will be as this size
    tempBuf.append([0]*seq_length) 
    seqs = tokenizer.texts_to_sequences(tempBuf)
    seqs = pad_sequences(seqs, maxlen=seq_length)
    i = 0
    for key, value in dataContent.items():
        dataContent[key] = seqs[i]
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

def createBaseModel(numWords, embeddingMatrix):
    ##--- Sub network for Bug Summary ---##
    inputBS = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
    tempTensorBS = Embedding(numWords, EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_BUG_SUMMARY_SEQUENCE_LENGTH, trainable=False) (inputBS)
    #tempTensorBS = Dropout(0.5)(tempTensorBS)
    tempTensorBS = Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))(tempTensorBS)
    tempTensorBS = Dropout(0.5)(tempTensorBS)
    featureBS = Dense(128, activation='relu')(tempTensorBS)
    ##-----------------------------------##

    ##--- Sub network for Bug Description ---##
    inputBD = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32')
    embeddedBD = Embedding(numWords, EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_BUG_DESC_SEQUENCE_LENGTH, trainable=False) (inputBD)
    convF1BD = Dropout(0.1)(embeddedBD)
    convF1BD = Conv1D(128, 3, activation='relu')(convF1BD)
    convF1BD = MaxPooling1D()(convF1BD)
    
    convF2BD = Dropout(0.1)(embeddedBD)
    convF2BD = Conv1D(128, 4, activation='relu')(convF2BD)
    convF2BD = MaxPooling1D()(convF2BD)    

    convF3BD = Dropout(0.1)(embeddedBD)
    convF3BD = Conv1D(128, 5, activation='relu')(convF3BD)
    convF3BD = MaxPooling1D()(convF3BD)    
    
    concatedBD = Concatenate(axis=1)([convF1BD, convF2BD, convF3BD]) 
    #tempTensorBD = GlobalMaxPooling1D()(concatedBD) # May replace with flatten
    tempTensorBD = Flatten()(concatedBD)
    tempTensorBD = Dropout(0.2)(tempTensorBD)
    featureBD = Dense(128, activation='relu')(tempTensorBD)
    ##---------------------------------------##
    
    if INPUT_MODE == 'SUM':
        model = Model(inputBS, featureBS) 
    elif INPUT_MODE == 'SUM+DESC':
        featureBA_BD = Concatenate()([featureBS, featureBD])
        model = Model([inputBS, inputBD], featureBA_BD) 
    else:
        print ("Input Mode not defined!")
        sys.exit(1)

    model.summary()
    return model 

def createTripletModel(baseModel):
    if INPUT_MODE == 'SUM': 
        inputA_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32') 
        inputAP_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        inputAN_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        featureA = baseModel(inputA_SUM)
        featureAP = baseModel(inputAP_SUM)
        featureAN = baseModel(inputAN_SUM)
        distance = Lambda(tripletLoss, output_shape=tripletOutputShape)([featureA, featureAP, featureAN])
        tripletModel = Model(inputs=[inputA_SUM, inputAP_SUM, inputAN_SUM], outputs=distance)
    elif INPUT_MODE == 'SUM+DESC':
        inputA_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32') 
        inputAP_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        inputAN_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        inputA_DESC  = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32') 
        inputAP_DESC = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32') 
        inputAN_DESC = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32') 
        featureA  = baseModel([inputA_SUM, inputA_DESC])
        featureAP = baseModel([inputAP_SUM, inputAP_DESC])
        featureAN = baseModel([inputAN_SUM, inputAN_DESC])
        distance = Lambda(tripletLoss, output_shape=tripletOutputShape)([featureA, featureAP, featureAN])
        tripletModel = Model(inputs=[inputA_SUM, inputA_DESC, inputAP_SUM, inputAP_DESC, inputAN_SUM, inputAN_DESC], outputs=distance)
    else:
        print ("Input Mode not defined!")
        sys.exit(1)

    tripletModel.summary()
    return tripletModel

def createClsModel(baseModel):
    if INPUT_MODE == 'SUM': 
        inputA = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32') 
        inputB = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        featureA = baseModel(inputA)
        featureB = baseModel(inputB)
    elif INPUT_MODE == 'SUM+DESC':
        inputA_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        inputA_DESC = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32')
        inputB_SUM = Input(shape=(MAX_BUG_SUMMARY_SEQUENCE_LENGTH,), dtype='int32')
        inputB_DESC = Input(shape=(MAX_BUG_DESC_SEQUENCE_LENGTH,), dtype='int32')
        featureA = baseModel([inputA_SUM, inputA_DESC])
        featureB = baseModel([inputA_SUM, inputA_DESC])
    else:
        print ("Input Mode not defined!")
        sys.exit(1)

    mergedFeature = Concatenate()([featureA, featureB])
    mergedFeature = Dropout(0.3)(mergedFeature)
    x = Dense(128, activation='relu')(mergedFeature)
    x = Dropout(0.3)(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    pred = Dense(1, activation='sigmoid')(x)

    if INPUT_MODE == 'SUM': 
        clsModel = Model(inputs=[inputA, inputB], outputs=pred)
    elif INPUT_MODE == 'SUM+DESC':
        clsModel = Model(inputs=[inputA_SUM, inputA_DESC, inputB_SUM, inputB_DESC], outputs=pred)
    else:
        print ("Input Mode not defined!")
        sys.exit(1)
 
    clsModel.summary()
    return clsModel


def genTripletsData(dataList, dataContentSum, dataContentDesc):
    triplets = []
    for dataSlot in dataList:
        bugA, bugAP, bugAN = dataSlot
        if INPUT_MODE == 'SUM': 
            triplets.append([dataContentSum[bugA], 
                             dataContentSum[bugAP], 
                             dataContentSum[bugAN]])
        elif INPUT_MODE == 'SUM+DESC':
            triplets.append([[dataContentSum[bugA],  dataContentDesc[bugA]], 
                             [dataContentSum[bugAP], dataContentDesc[bugAP]], 
                             [dataContentSum[bugAN], dataContentDesc[bugAN]]])
        else:
            print ("Input Mode not defined!")
            sys.exit(1)
    return triplets       

def createDataInputForBaseModel(tripletsData):
    if INPUT_MODE == 'SUM':
        A=[]
        AP=[]
        AN=[]
        for item in tripletsData:
            A.append(item[0])
            AP.append(item[1])
            AN.append(item[2])
        combinedInput = [A, AP, AN] 
    elif INPUT_MODE == 'SUM+DESC':
        A_SUM=[]
        AP_SUM=[]
        AN_SUM=[]
        A_DESC=[]
        AP_DESC=[]
        AN_DESC=[]
        for item in tripletsData:
            A_SUM.append((item[0])[0])
            A_DESC.append((item[0])[1])
            AP_SUM.append((item[1])[0])
            AP_DESC.append((item[1])[1])
            AN_SUM.append((item[2])[0])
            AN_DESC.append((item[2])[1])
        combinedInput = [A_SUM, A_DESC, AP_SUM, AP_DESC, AN_SUM, AN_DESC] 
    else:
        print ("Input Mode not defined!")
        sys.exit(1)
    return combinedInput

def createDataInputForClsModel(tripletsData):
    if INPUT_MODE == 'SUM':
        A = []
        B = []
        Label = []
        for item in tripletsData:
            A.append(item[0])
            B.append(item[1])
            Label.append(1)
            A.append(item[0])
            B.append(item[2])
            Label.append(0)
        combinedInput = [A, B] 
    elif INPUT_MODE == 'SUM+DESC':
        A_SUM=[]
        B_SUM=[]
        A_DESC=[]
        B_DESC=[]
        Label = []
        for item in tripletsData:
            A_SUM.append((item[0])[0])
            A_DESC.append((item[0])[1])
            B_SUM.append((item[1])[0])
            B_DESC.append((item[1])[1])
            Label.append(1)
            A_SUM.append((item[0])[0])
            A_DESC.append((item[0])[1])
            B_SUM.append((item[2])[0])
            B_DESC.append((item[2])[1])
            Label.append(0)
        combinedInput = [A_SUM, A_DESC, B_SUM, B_DESC] 
    else:
        print ("Input Mode not defined!")
        sys.exit(1)
    return combinedInput, Label
####################################

###==================== Main ====================###
ts = time.localtime(time.time())
tsStr = str(ts.tm_year)+str(ts.tm_mon)+str(ts.tm_mday)+str(ts.tm_hour)+str(ts.tm_min)+str(ts.tm_sec)
resultPath = os.path.join(RESULT_PATH, tsStr)
os.mkdir(resultPath)
os.system("cp *.py "+resultPath) ## save training scripts

############### data setup #################
trainDataList, trainDataIndex = loadDataList(os.path.join(BASE_PATH, TRAIN_SET_FILE))
valDataList, valDataIndex = loadDataList(os.path.join(BASE_PATH, VAL_SET_FILE))

trainDataContentSummary, trainDataContentDesc = loadData(trainDataIndex)
valDataContentSummary, valDataContentDesc = loadData(valDataIndex)

with open(os.path.join(BASE_PATH, TOKENIZER_FILE_NAME), 'rb') as handle:
    tokenizer = pickle.load(handle)

convertDataVector(trainDataContentSummary, MAX_BUG_SUMMARY_SEQUENCE_LENGTH, tokenizer)
if INPUT_MODE == 'SUM+DESC':
    convertDataVector(trainDataContentDesc, MAX_BUG_DESC_SEQUENCE_LENGTH, tokenizer)

convertDataVector(valDataContentSummary, MAX_BUG_SUMMARY_SEQUENCE_LENGTH, tokenizer)
if INPUT_MODE == 'SUM+DESC':
    convertDataVector(valDataContentDesc, MAX_BUG_DESC_SEQUENCE_LENGTH, tokenizer)

embeddingsIndex = buildWE()
embeddingMatrix, numWords = genEmbeddingMatrix(tokenizer.word_index, embeddingsIndex)

trainTriplets = genTripletsData(trainDataList, trainDataContentSummary, trainDataContentDesc)
valTriplets = genTripletsData(valDataList, valDataContentSummary, valDataContentDesc)
############################################# 

############# create base Model #############
baseModel = createBaseModel(numWords, embeddingMatrix) 
#############################################

########### create triplet Model ############
tripletModel = createTripletModel(baseModel)
#############################################

##### prepare dataset for triplet model #####
combinedTrainInput = createDataInputForBaseModel(trainTriplets)
combinedValInput = createDataInputForBaseModel(valTriplets)
#############################################

############ train triplet model ############
baseFilePath = os.path.join(resultPath, "base-{epoch:02d}-{val_loss:.2f}.h5")
baseCP = ModelCheckpoint(baseFilePath, verbose=1, period=1)
baseCallbacks = [baseCP]
tripletModel.compile(loss=identicalLoss, optimizer=optimizers.RMSprop(lr=1e-4), metrics=None)
tripletModel.fit(combinedTrainInput,
          np.ones(len(trainTriplets)).tolist(),
          batch_size=BASE_MODEL_BATCH_SIZE,
          epochs=BASE_MODEL_EPOCHS,
          validation_data=(combinedValInput, np.ones(len(valTriplets)).tolist()),
          callbacks=baseCallbacks)
baseModel.save(os.path.join(resultPath, "infoRetrieval.h5"))
#############################################

############ create cls model ###############
clsModel = createClsModel(baseModel)
#############################################

###### prepare dataset for cls model ########
combinedClsTrainInput, clsTrainLabel = createDataInputForClsModel(trainTriplets)
combinedClsValInput, clsValLabel = createDataInputForClsModel(valTriplets)
#############################################

############# train cls model ###############
allFilePath = os.path.join(resultPath, "all-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5")
allCP = ModelCheckpoint(allFilePath, verbose=1, period=1)
allCallbacks = [allCP]
clsModel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
clsModel.fit(combinedClsTrainInput,
             clsTrainLabel,
             batch_size=ALL_MODEL_BATCH_SIZE,
             epochs=ALL_MODEL_EPOCHS,
             validation_data=(combinedClsValInput, clsValLabel),
             callbacks=allCallbacks)
#############################################
###==============================================###
