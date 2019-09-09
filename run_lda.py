# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:43:33 2019

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:27:24 2019

@author: dell
"""

import sys
import argparse
import json
import numpy as np
from LDA import lda_model, corp_dict
import random as rd
from gensim.models import CoherenceModel

if __name__=='__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-k","--k",type = int,default = 5)#topic number
     parser.add_argument("-tf","--tfidf",type = bool,default = True)
     parser.add_argument("-tr","--train",type = bool,default = True)# whether or not select model
     parser.add_argument("-ts",'--tsne',type = bool,default = True)# whether or not use tsne 
     
     
     '''
     relative loose parameters #for lda model (gensim)
     '''
     parser.add_argument("-cksz","--chunksize",type = int,default = 32)
     parser.add_argument("-ps","--passes",type = int,default = 10)
     parser.add_argument("-ite","--iteration",type = int,default = 5)
     parser.add_argument("-db","--dictionary_below",type = int,default = 10)
     parser.add_argument("-da","--dictionary_above",type = float,default = 0.9)
     #parser.add_argument("-wks","--workers",type = int,default = 3) #parrallized
     parser.add_argument("-al","--alpha",type = str,default = 'asymmetric')
     parser.add_argument("-dc","--decay",type = float,default = 0.5)
     
     args = parser.parse_args()
     
     print('Get dic and corpus!')
     cp_dic = corp_dict(tf_idf = args.tfidf,dic_below = args.dictionary_below,dic_above = args.dictionary_above)
     corpus = cp_dic.corpus
     dictionary = cp_dic.dictionary     
     processed_docs = cp_dic.processed_docs
     inp = open('test.json','rb')
     data = json.load(inp)
     inp.close()
     
     def train_model():
         print('choose topics!')
         top_lst = list(range(2,11)) + list(range(12,20,2)) + list(range(20,100,10))
         tfidf_v = [True,False]
         min_prep = 10000000#init
         min_k=-1
         min_tfidf = None
         for tf_idf in tfidf_v:
             for k in top_lst:
                 print(k)
                 train_idx = rd.sample(range(len(corpus)),int(0.9*len(corpus)))
                 test_idx = list(set(range(len(corpus))).difference(set(train_idx)))
                 train_corp = cp_dic.get_extra(np.array(processed_docs)[train_idx],tf_idf)
                 test_corp = cp_dic.get_extra(np.array(processed_docs)[test_idx],tf_idf)
                 
                 _lda_model = lda_model(topic_num=k,corpus=train_corp,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=tf_idf,decay = args.decay)
                 cur_prep = _lda_model.get_prep(test_corp)
                 if cur_prep < min_prep:
                     min_k,min_tfidf = k,tf_idf
                 print('topic:{0}--tf_idf{1}->prep:{2}'.format(k,tf_idf,cur_prep))
         _lda_model = lda_model(topic_num=min_k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=min_tfidf,decay = args.decay)
         
         _lda_model.save_model()
         return _lda_model
     
     if args.train:
         _lda_model = train_model()
         _lda_model.tsne_vis(data)
         _lda_model.lda_vis(corpus=corpus,dictionary=dictionary)
     else:
         _lda_model = lda_model(topic_num=args.k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=args.tf_idf,decay = args.decay)
         _lda_model.show_lda()
         _lda_model.tsne_vis(data)
         _lda_model.lda_vis(corpus = corpus,dictionary = dictionary)
         
     

        
      
        
        
     
        
        
        
         
         