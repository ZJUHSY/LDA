# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:02:05 2019

@author: dell
"""
import nltk
import nltk.data
#nltk.download()
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')#for tokenize
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import WordPunctTokenizer  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer#词形归一
from nltk.stem import PorterStemmer 
#from nltk.tokenize import word_tokenize 
import re

#define 2 tokenize and stopwords
ps = PorterStemmer() 
wordnet_lemmatizer = WordNetLemmatizer()
stop_lst = stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-'] #去除标点符号

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))

def check_word(word):
    word= word.lower()
    if word in stop_lst or word in english_punctuations:
        return False
    elif isSymbol(word):
        return False
    elif word.isnumeric():
        return False
    else:
        return True

    
def wordtokenizer(sentence): #word spliter and remove stopwords
    #分段
    words = WordPunctTokenizer().tokenize(sentence) #nltk.word_tokenizer()
    #print(type(words))
    filtered = [ps.stem(wordnet_lemmatizer.lemmatize(w)) for w in words if check_word(w)]
    #filtered = [ps.stem(w) for w in filtered if check_word(w)]
    Rfiltered =nltk.pos_tag(filtered)
    return [x[0] for x in Rfiltered] # remove pos



