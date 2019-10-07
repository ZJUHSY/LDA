import argparse
import json
import numpy as np
try:
    from .LDA import lda_model
except:
    from LDA import lda_model
import gensim
import pickle
import random as rd
from gensim.models import CoherenceModel
import pandas as pd
import os
from sklearn.manifold import TSNE
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer#词形归一
from nltk.stem import PorterStemmer
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')#for tokenize
# nltk.download('stopwords')
# nltk.download('wordnet')
ps = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
# get processed docs
def get_processed_docs():
    _inp = open('pro_docs.json', 'rb')
    processed_docs = json.load(_inp)
    _inp.close()
    return processed_docs


# get dictionary
def get_dic():
    return gensim.corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), 'dictionary.gensim'))


# get corpus
def get_corp(tf_idf = False):
    if tf_idf:

        #corpus = pickle.load(open('corpus.pkl_tfidf','rb'))
        corpus = pickle.load(open(os.path.join(os.path.dirname(__file__), 'corpus.pkl_tfidf'), 'rb'))
    else:

        #corpus = pickle.load(open('corpus.pkl','rb'))
        corpus = pickle.load(open(os.path.join(os.path.dirname(__file__), 'corpus.pkl'), 'rb'))
    return corpus


def get_model(corpus, dic, k, ck_size=32, ps=10, ite=5, decay=0.5):  # k is the topic number

    _lda_model = lda_model(topic_num=k, corpus=corpus, dictionary=dic, ite=ite, ps=ps, alpha='asymmetric',
                           ck_size=ck_size, decay=decay, path = 'lda_model' + str(k))
    _lda_model.save_model()
    return _lda_model


# input
def get_topic_weight(k,
                     sel_idx=[]):  # output: 1. get the per-document topic vector 2. get each document's most relevant topic index
    dictionary = get_dic()
    corpus = get_corp()
    model = get_model(k=k, corpus=corpus, dic=dictionary)
    _lda_model  = model.get_model()
    if len(sel_idx) != 0:
        corpus = list(np.array(corpus)[sel_idx])

    doc_topic_mat = _lda_model[corpus]
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

def tsne_coord(doc_topic_weight):  # k is the topic number  // return: tsne coordinates
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(doc_topic_weight)
    return tsne_lda

def save_tsne_coord(k, coord, topic_arr):
    _coord_name = 'coord' + str(k) +'.txt'
    _topic_name = 'topic_arr' + str(k) + '.txt'
    save_path1 = os.path.join(os.path.dirname(__file__), 'tsne', _coord_name)
    save_path2 = os.path.join(os.path.dirname(__file__), 'tsne', _topic_name)
    np.savetxt(save_path1, coord)
    np.savetxt(save_path2, topic_arr)


def LDA_vis(k, sel_idx = []):
    #k: number of topics, sel_idx: index to choose  #output: html
    dictionary = get_dic()
    corpus = get_corp()
    model = get_model(corpus=corpus, dic=dictionary, k=k)
    model.lda_vis(sel_idx)

def wc(k, sel_idx = []):  #produce png saved wordcloud
    dictionary = get_dic()
    corpus = get_corp()
    model = get_model(corpus=corpus, dic=dictionary, k=k)
    model.wordcloud_topic(sel_idx)

def topic_summaries(k, model):   #k: topic number model: LDA model
    topic_summaries = []
    for i in range(k):
        keys = [x[0] for x in model.show_topic(i, topn=5)]
        topic_summaries.append(' '.join(keys))
    return topic_summaries


def save_model(model): #input: LDA_model
    model.save_model()




# k_lst = range(4,9)
# for k in k_lst:
#     corpus = get_corp()
#     dic = get_dic()
#     model = get_model(corpus, dic, k)
#     _lda_model = model.get_model()
#     summaries = topic_summaries(k, model=_lda_model)
#     save_name = 'topic_summaires' + str(k) + '.txt'
#     outp = open(os.path.join(os.path.dirname(__file__), 'summaries', save_name), 'w')
#     outp.write(str(summaries))
#     outp.close()

# for k in k_ls
#     topic_weight, topic_arr = get_topic_weight(k)
#     coord = tsne_coord(topic_weight)
#     save_tsne_coord(k, coord, topic_arr)

from translate import Translator
_inp = open('pro_docs.json', 'rb')
processed_docs = json.load(_inp)
_inp.close()
translator = Translator(from_lang='chinese', to_lang='english')
for doc in processed_docs:
    for idx, word in enumerate(doc):
        print(word)
        if word.encode('utf-8').isalnum():
            doc[idx] = ps.stem(wordnet_lemmatizer.lemmatize(word))
            continue
        else:
            try:
                res = translator.translate(word)
                doc[idx] = res
                print(res)
            except:
                print('original')
outp = open('en_pro_docs.json', 'w')
outp.write(json.dumps(processed_docs))
outp.close()
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.9, keep_n=80000)
dictionary.save('en_dictionary.gensim')
corpus = get_corp()
dic = get_dic()
model = get_model(corpus, dic, 5)
model.lda_vis(en=True)