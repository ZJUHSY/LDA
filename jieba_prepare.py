# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:06:28 2019

@author: dell
"""

import jieba
import numpy as np
import json

class jb_cut():
    
    def __init__(self,path = 'test.json',add_file = 'blockchain_dic.txt',stop_file = 'stopwords-master/哈工大停用词表.txt'):
        inp = open(path,'rb')
        self.passages = json.load(inp)
        inp.close()
        
        #initialize members
        self.processed_docs = []
        
        #加入关键词
        jieba.add_word('比特币')
        jieba.add_word('区块链')
        jieba.add_word('以太坊')
        jieba.add_word('数字货币')
        jieba.add_word('去中心化')
        jieba.add_word('云计算')
        jieba.add_word('做市')
        jieba.add_word('莱特币')
        jieba.add_word('瑞波币')
        jieba.add_word('柚子币')
        jieba.add_word('okex')
        jieba.add_word('火币网')
        jieba.add_word('OKCoin')
        jieba.add_word('人工智能')
        jieba.add_word('云技术')
        jieba.add_word('大数据')
        jieba.add_word('加密账本')
        jieba.add_word('九个亿')
        #加入关键词
        add_word_list = []
        add_f = open(add_file, 'r',encoding='utf-8')
        add_obj = add_f.read()
        add_f.close()
        
        add_word_list = add_obj.split(' ')
        for add_word in add_word_list[:]:
            jieba.add_word(add_word)
        #jieba.add_word('9个亿')
        
        #del stop words and unused words
        del_list = [',','，','.','。','?','？','!','！' ,':','：',';','；','[','{',']','}','【','】','#',')','(','（','）','`','~'
           ,'·','@','/','、','九个亿','%','根据','date','passage','算出','交于','独家分析','规程','费是','亚于','非关键','关键', 
        '多于','少于','是','不是','差不多','约等于','可能','不可能',' 超出',' 下降',' 上升','  ',' |',' |',' +',' -',' *',' &',' ^',
           '$',' 一',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9',' 0',' >', ' <',' >=',' <=',' =',' 》',' 《',' "',' ”',' “',' 到',' 了',' 头',
           '如果',' 但是',' 又',' 既',' title','#',' 的',' 与',' 或',' \n',' 已',' 已经',' 根据',' 据',' 被',' 和',' 就',' 目前',' 有',' 就是',
        '从',' 都',' 得',' 地',' 着',' ......',' 这',' 那',' 91',' 独家分析',' 报道',' 分析',' 独家',' 不代表',' 观点']#输出词库
        #去除空格
        for i in range(0,len(del_list)):
            del_list[i] = del_list[i].strip()
        
        del_list.append('\n')
        del_list.append('')
        del_list.append(' ')
        del_list.append('#')
        del_list.append('将')
        del_list.append('个')
        del_list.append('日')
        del_list.append('月')
        del_list.append('一个')
        del_list.append('不')
        del_list.append('也')
        del_list.append('但')
        del_list.append('新')
        del_list.append('使得')
        del_list.append('使')
        del_list.append('人')
        del_list.append('个')
        del_list.append('为')
        del_list.append('等')
        del_list.append('其')
        del_list.append('改')
        del_list.append('年')
        del_list.append('上')
        del_list.append('下')
        del_list.append('会')
        del_list.append('在')
        del_list.append('更')
        del_list.append('让')
        del_list.append('所以')
        del_list.append('让')
        del_list.append('做')
        del_list.append('要')
        del_list.append('需要')
        del_list.append('这些')
        #filter stop words
        stop_f = open('stopwords-master/哈工大停用词表.txt','r',encoding = 'utf-8')
        stop_obj = stop_f.read()
        stop_obj_lst = stop_obj.split()
        del_list.extend(stop_obj_lst)
        stop_f.close()
        
        self.del_list = list(np.unique(np.array(del_list)))
        
        
    def cut_words(self,passage):
        jb_tmp = jieba.cut(passage,cut_all = False)
        tmp = '/'.join(jb_tmp)
        word_list = tmp.split('/')
        res = []
        for word in word_list:
            if word in self.del_list or len(word)<3 or word.isnumeric(): #del prue numebers
                continue
            if word.isalpha():
                word = word.lower()
            res.append(word)
        return res
    
    def process(self):
        processed_docs = [] # store tokenized words
        index = 0
        for passage in self.passages:
            if index%1000==0:
                print(index)
            res = self.cut_words(passage['passage'])
            processed_docs.append(res)
            index += 1
        #self.processed_docs = processed_docs
        return processed_docs
        #save data
        
        
        