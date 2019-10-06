import argparse
import json
import numpy as np
from LDA import lda_model
import gensim
import pickle
import random as rd
from gensim.models import CoherenceModel
import pandas as pd


# get processed docs
def get_processed_docs():
    _inp = open('pro_docs.json', 'rb')
    processed_docs = json.load(_inp)
    _inp.close()
    return processed_docs


# get dictionary
def get_dic():
    return gensim.corpora.Dictionary.load('dictionary.gensim')


# get corpus
def get_corp(tf_idf):
    if tf_idf:
        corpus = pickle.load(open('corpus.pkl_tfidf', 'rb'))
    else:
        corpus = pickle.load(open('corpus.pkl', 'rb'))
    return corpus


def get_model(corpus, dic, k, ck_size=32, ps=10, ite=5, decay=0.5):  # k is the topic number
    _lda_model = lda_model(topic_num=k, corpus=corpus, dictionary=dic, ite=ite, ps=ps,
                           ck_size=ck_size, decay=decay)
    return _lda_model


# input
def get_topic_weight(k,
                     sel_idx=[]):  # output: 1. get the per-document topic vector 2. get each document's most relevant topic index
    dictioary = get_dic()
    corpus = get_corp()
    model = get_model(topic_num=k, corpus=corpus, dictionary=dictioary)

    if len(sel_idx) != 0:
        corpus = list(np.array(corpus)[sel_idx])

    doc_topic_mat = model[corpus]
    doc_topic_weight = []
    for doc in doc_topic_mat:
        # print(doc)
        if isinstance(doc, tuple):
            doc = [doc]  # change to list
        arr = np.zeros(k)
        topic_weight = [x[1] for x in doc]
        if len(topic_weight) < k:
            topics = [x[0] for x in doc]
            arr[topics] = topic_weight
        else:
            arr = np.array(topic_weight)
        doc_topic_weight.append(arr)
    doc_topic_weight = np.array(doc_topic_weight)  # [_idx]#get per-doc topic weight matrix for tsne
    topic_arr = np.argmax(doc_topic_weight, axis=1)  # every most document's most related topic
    return doc_topic_weight, topic_arr
