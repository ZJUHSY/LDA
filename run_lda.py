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

if __name__=='__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-k","--k",type = int,default = 5)#topic number
     parser.add_argument("-tf","--tfidf",type = bool,default = True)
     
     
     '''
     relative loose parameters #for lda model (gensim)
     '''
     parser.add_argument("-cksz","--chunksize",type = int,default = 32)
     parser.add_argument("-ps","--passes",type = int,default = 10)
     parser.add_argument("-ite","--iteration",type = int,default = 5)
     parser.add_argument("-db","--dictionary_below",type = int,default = 10)
     parser.add_argument("-da","--dictionary_above",type = float,default = 0.9)
     parser.add_argument("-wks","--workers",type = int,default = 3) #parrallized
     parser.add_argument("-al","--alpha",type = str,default = 'asymmetric')
     
     
     args = parser.parse_args()
     
     print('Get dic and corpus!')
     cp_dic = corp_dict(tf_idf = args.tfidf,dic_below = args.dictionary_below,dic_above = args.dictionary_above)
     corpus = cp_dic.corpus
     dictionary = cp_dic.dictionary     
     
     print('choose topics!')
     top_lst = list(range(2,11)) + list(range(12,21,2)) # trying for the best topics
     min_perp = 10000000#init
     min_k=-1
     for k in top_lst:
         _lda_model = lda_model(topic_num=k,corpus=corpus,dictionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha,tf_idf=args.tf_idf)
         cur_prep = _lda_model.get_prep()
         print(type(cur_prep))
         print('topic:{0} prep:{1}'.format(k,cur_prep))
         
         if cur_prep<min_perp:
             min_perp = cur_prep
             min_k = k
     #find the topic model with the smallest perp
     _lda_model = lda_model(topic_num=min_k,corpus=corpus,dcitionary=dictionary,ite=args.iteration,ps=args.passes,
                               ck_size=args.chunksize,alpha=args.alpha)
     _lda_model.save_model()
     
     
         
     
     
     
     