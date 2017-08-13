#encoding:utf-8
import re
import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
import codecs
import sys
np.random.seed(0)

def getEmbedding(infile_path, char2id):
    row_index = 0
    count = 0
    infile = codecs.open(infile_path, "r", 'utf-8')
    for row in infile:
        items = row.strip().split()
        row_index += 1
        if row_index == 1:
            num_chars = int(items[0])
            emb_dim = int(items[1])
            #emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
            emb_matrix = np.random.normal(loc=0.0, scale=1.0, size=(len(char2id.keys()), emb_dim))
            continue
        char = items[0]
        emb_vec = [float(val) for val in items[1:]]
        if char in char2id:
            count += 1
            emb_matrix[char2id[char]] = emb_vec
        if row_index % 10000 == 0:
            print "%s\r"%(row_index),
            
    print count, len(char2id.keys())
    emb_matrix[0] = np.zeros(emb_dim)
    
    return emb_matrix

def nextBatch(X, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch

def nextRandomBatch(X, y, batch_size=128):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch

# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
        else:
            sample[i] = sample[i][:seq_max_len]
    return sample

def loadMap(token2id_filepath):
    token2id = {}
    id2token = {}
    with codecs.open(token2id_filepath, 'r', 'utf8') as infile:
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token

def saveMap(id2char, id2label, model_path):
    with codecs.open(os.path.join(model_path, "char2id"), "w", "utf8") as outfile:
        for idx in id2char:
            outfile.write(str(id2char[idx]) + "\t" + str(idx)  + "\n")
    with codecs.open(os.path.join(model_path, "label2id"), "w", "utf8") as outfile:
        for idx in id2label:
            outfile.write(str(id2label[idx]) + "\t" + str(idx) + "\n")
    print "saved map between token and id"

def get_words(filepath):
    chars = set()
    labels = set()
    if not filepath:
        return char, labels
    infile = codecs.open(train_path, 'r', 'utf8')
    for line in infile:
        items = line.strip().split(' | ')
        labels.add(int(items[0]))
        chars.update(items[1].split(' '))
    return list(chars), list(labels)
  
def buildMap(train_path, model_path):
    chars, labels = get_words(train_path)
    labels = sorted(labels)
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label =  dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    saveMap(id2char, id2label, model_path)

    return char2id, id2char, label2id, id2label

def get_features(filepath, char2id, label2id, seq_max_len):
    def mapFunc(x, char2id):
        if x not in char2id:
            return char2id["<NEW>"]
        else:
            return char2id[x]
    
    infile = codecs.open(train_path, 'r', 'utf8')
    X = []
    Y = []
    for line in infile:
        y = [0] * len(label2id)
        x = []
        items = line.strip().split(' | ')
        y[label2id[items[0]]] = 1
        for word in items[1].split(' '):
            x.append(mapFunc(char2id[word], char2id))
        
    X = padding(X, seq_max_len)
    
    return X, Y

def getTrain(train_path, val_path, model_path, train_val_ratio, seq_max_len):
    char2id, id2char, label2id, id2label = buildMap(train_path, model_path)
     X, y = get_features(train_path, char2id, label2id, seq_max_len)
    
    num_samples = len(X)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    X = X[indexs]
    y = y[indexs]

    if val_path != None:
        X_train = X
        y_train = y
        X_val, y_val = getTest(val_path, model_path, seq_max_len)
    else:
        X_train = X[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        X_val = X[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

    print "train size: %d, validation size: %d" %(len(X_train), len(X_val))

    return X_train, y_train, X_val, y_val

def getTest(test_path, model_path, seq_max_len):
    char2id, id2char = loadMap(os.path.join(model_path, "char2id"))
    label2id, id2label = loadMap(os.path.join(model_path, "label2id"))
    X_test, y_test = get_features(test_path, char2id, label2id, seq_max_len)
    
    print "test size: %d" %(len(X_test), len(y_test))
    
    return X_test, y_test
    
if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf8")