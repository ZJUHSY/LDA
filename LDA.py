# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:36:42 2019

@author: dell
"""
#this file is added to produce corpora as well as LDA model
import pandas as pd
import numpy as np
import os
#load in/out data
import pickle
import json
#gensim packages
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pyLDAvis.gensim
#split word
from jieba_prepare import jb_cut
#worldcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


class corp_dict(): #prouce 
    def __init__(self,path = 'test.json',tf_idf = True,dic_below = 5,dic_above = 0.9,dic_keep = 80000,new = False): #tf_idf: whether or not use tf_idf method to produce
        if os.path.isfile('dictionary.gensim') and not new:#if new,corpus beside model should be loaded
            self.dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
            if tf_idf:
                self.corpus = pickle.load(open('corpus.pkl_tfidf', 'rb'))
            else:
                self.corpus = pickle.load(open('corpus.pkl', 'rb'))
            return
        else:
            #use jieba to produce word list 
            jb = jb_cut(path)
            processed_docs = jb.process()
            
            #use processed_docs to produce dictionary and corpus
            dictionary = Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=dic_below, no_above=dic_above,keep_n=dic_keep)
            self.dictionary = dictionary
            if not new:
               dictionary.save('dictionary.gensim')
            
            corpus = [dictionary.doc2bow(text) for text in processed_docs]
            if not new:    
                pickle.dump(corpus, open('corpus.pkl', 'wb'))
        
            if tf_idf:
                tfidf_model = models.TfidfModel(corpus)
                corpus = tfidf_model[corpus]
                if not new:
                    pickle.dump(corpus, open('corpus.pkl_tfidf', 'wb'))#save cirpus
            self.corpus = corpus
      
        
class lda_model():
    def __init__(self,topic_num,corpus,dictionary,ite,ps,ck_size,alpha,tf_idf = True): #decide topic num for LDA 
        self.model = gensim.models.LdaMulticore(corpus=corpus,num_topics=topic_num,id2word=dictionary,chunksize=ck_size,
                                                passes=ps,alpha=alpha,eval_every=1,iterations=ite)
        self.k = topic_num
        self.corpus = corpus
        self.dic = dictionary
        self.tf_idf = tf_idf

    def save_model(self):
        str_tfidf = ''
        if self.tf_idf:
            str_tfidf = ' tf_idf'
        save_name = 'lda_model' + str(self.k) + str_tfidf 
        #if not os.path.isfile(save_name):
        self.model.save(save_name)
    
        
    def show_lda(self):#print topics(word_weight)
        topics = self.model.print_topics(num_words=4)
        print(topics)
        
    '''
     corpus and dic mgiht be new
    '''    
    def lda_vis(self,corpus,dictionary):#use pyLDAVIS produce html to show
        lda_display = pyLDAvis.gensim.prepare(self.model, corpus, dictionary, sort_topics=False)
        save_name = 'lda' + str(self.k) + '.html'
        if not os.path.isfile(save_name):
            pyLDAvis.save_html(lda_display, save_name)
    
    def get_prep(self):
        #return self.model.bound(self.corpus)
        return self.model.log_perplexity(self.corpus)
    
    def new_inference(self,other_corpus,update =False): #doing new inference or update lda model at the same time
        res = []
        for doc in other_corpus:
            res.append(self.model[doc])#add topic vec
        if update: #use unseen doc to update lda model
            self.model.update(other_corpus)
        return res
    
    def tsne_vis(self):
        pass
    
#    def wordcloud_topic(self):
#        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
#        cloud = WordCloud(stopwords=stop_words,
#                          background_color='white',
#                          width=2500,
#                          height=1800,
#                          max_words=10,
#                          colormap='tab10',
#                          color_func=lambda *args, **kwargs: cols[i],
#                          prefer_horizontal=1.0)
#        
#        fig, axes = plt.subplots(2, 4, figsize=(10,10), sharex=True, sharey=True)
#        
#        for i, ax in enumerate(axes.flatten()):
#            fig.add_subplot(ax)
#            topic_words = dict(topics[i][1])
#            cloud.generate_from_frequencies(topic_words, max_font_size=300)
#            plt.gca().imshow(cloud)
#            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
#            plt.gca().axis('off')
#        
#        
#        plt.subplots_adjust(wspace=0, hspace=0)
#        plt.axis('off')
#        plt.margins(x=0, y=0)
#        plt.tight_layout()
#        plt.show()
    
    
    #def show_docs()
        
        
            
            
            
        
        
                                            
         