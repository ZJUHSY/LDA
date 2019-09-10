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
import random as rd
#gensim packages
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim import models#,corpora
import pyLDAvis.gensim
#split word
from jieba_prepare import jb_cut
#worldcloud
#from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
#import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

#bokeh tools
#from bokeh.plotting import figure, output_file, show
#from bokeh.models import Label
#import matplotlib.colors as mcolors
#from bokeh.io import output_notebook
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool


class corp_dict(): #prouce 
    def __init__(self,path = 'test.json',tf_idf = True,dic_below = 5,dic_above = 0.9,dic_keep = 80000,new = False): #tf_idf: whether or not use tf_idf method to produce
        if os.path.isfile('dictionary.gensim') and not new:#if new,corpus beside model should be loaded
            #load data
            inp = open(path,'rb')
            self.data = pd.DataFrame(json.load(inp))
            inp.close()
            _inp = open('pro_docs.json','rb')
            self.processed_docs = json.load(_inp)
            _inp.close()
            
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
            self.processed_docs = processed_docs #used for train
            
            outp = open("pro_docs.json", 'w', encoding="utf-8")
            outp.write(json.dumps(self.processed_docs, indent=4, ensure_ascii=False))
            outp.close()
            
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

    def get_extra(self,ex_data,tf_idf): #use processed_docs to get subset corpus/ for train
        #dic = Dictionary(ex_data)
        #dic.filter_extremes(no_below = 5,no_above = 0.9,keep_n = 80000)
        other_corpus = [self.dictionary.doc2bow(text) for text in ex_data]
        if tf_idf:
            tfidf_model = models.TfidfModel(other_corpus)
            other_corpus = tfidf_model[other_corpus]
        return other_corpus
        
        
        
        
class lda_model():
    def __init__(self,topic_num,corpus,dictionary,ite,ps,ck_size,alpha,decay,tf_idf = True): #decide topic num for LDA 
        self.model = gensim.models.LdaMulticore(corpus=corpus,num_topics=topic_num,id2word=dictionary,chunksize=ck_size,
                                                passes=ps,alpha=alpha,eval_every=1,iterations=ite,decay=decay)
        self.k = topic_num
        self.corpus = corpus
        self.dic = dictionary
        self.tf_idf = tf_idf

    def save_model(self):
        str_tfidf = ''
        if self.tf_idf:
            str_tfidf = '-tf_idf'
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
    
    def get_prep(self,corpus):
        #return self.model.bound(self.corpus)
        return self.model.log_perplexity(corpus)
    
    def new_inference(self,other_text,update =False): #doing new inference or update lda model at the same time
        res = []
        other_corpus = [self.dic.doc2bow(word) for word in other_text]
        for doc in other_corpus:
            res.append(self.model[doc])#add topic vec
        if update: #use unseen doc to update lda model
            self.model.update(other_corpus) #other corpus need to have not other dictionary
        return res
    
    
    #visualize per-document tsne projection 
    ### output: html representaions
    def tsne_vis(self,data,threshold = 0.5,topn = 5): #data is the source of the data
        #extract matrix
        doc_topic_mat = self.model[self.corpus]
        doc_topic_weight = []
        for doc in doc_topic_mat:
            arr = np.zeros(self.k)
            topic_weight = [x[1] for x in doc]
            if len(topic_weight) < self.k:
                topics = [x[0] for x in doc]
                arr[topics] = topic_weight
            else:
                arr = topic_weight
            doc_topic_weight.append(arr)
       
        
        #extract ooint to reduce perplexity
        #threshold = 0.5
        _idx = np.amax(doc_topic_weight, axis=1) > threshold  # idx of doc that above the threshold
        self.doc_topic_weight = np.array(doc_topic_weight)[_idx]#get per-doc topic weight matrix for tsne
        self.topic_arr = np.argmax(self.doc_topic_weight, axis=1)#every most document's most related topic
        
        ###running tsne
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        self.tsne_lda = tsne_model.fit_transform(self.doc_topic_weight)
        
        
        '''
        prepare souce data for bokeh
        '''
        source_data = pd.DataFrame(columns = ['x_values','y_values',
                                      'color','content','topic','semantic','label'])
        source_data['x_values'] = self.tsne_lda[:,0]
        source_data['y_values'] = self.tsne_lda[:,1]
        #mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
        #source_data['color'] = mycolors[self.topic_arr]
        color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(self.k)]
        source_data['color'] = np.array(color)[self.topic_arr]
        source_data['content'] = data['passage'].values[_idx]
        source_data['topic'] = self.topic_arr
        source_data['semantic'] = data['semantic_value'].values[_idx]
        source_data['label'] = data['label'].values[_idx]
        
        #get each topic's key word to visualize
        topic_summaries = []
        for i in range(self.k):
            keys = [x[0] for x in self.model.show_topic(i,topn=topn)]
            topic_summaries.append(' '.join(keys))
        
            
        
        #bokeh visualize
        str_tfidf = ''
        if self.tf_idf:
            str_tfidf = '-tfidf'
        title = 'per-document-tsne-vis' + 'lda_model' + str(self.k) + str_tfidf 
        #num_example = self.doc_topic_weight.shape[0]
        
        plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)
        
        plot_lda.scatter(x='x_values', y='y_values',color = 'color',
                 source=bp.ColumnDataSource(source_data))
        
        #add text
        topic_coord = np.empty((self.doc_topic_weight.shape[1], 2)) * np.nan
        for topic_num in self.topic_arr:
          if not np.isnan(topic_coord).any():
            break
          topic_coord[topic_num] = self.tsne_lda[list(self.topic_arr).index(topic_num)]
        
        # plot crucial words
        for i in range(self.doc_topic_weight.shape[1]):
          plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
        
        # hover tools
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key - semantic: @semantic - label: @label"}
        
        # save the plot
        save(plot_lda, '{}.html'.format(title))
        
        
    
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
        
        
            
            
            
        
        
                                            
         