
# coding: utf-8

# In[87]:

import numpy as np
import sys
import os
import string
import source
import math


# In[99]:

class Vector:
    def __init__(self,tfs,threshold):
        self.tfs = tfs
        #self.filter_freqs(threshold)
        self.normalize()
    def filter_freqs(self,t):
        for tf in list(self.tfs.keys()):
            if self.tfs[tf] > t:
                del self.tfs[tf]
    def normalize(self):
        max_val = np.max(list(self.tfs.values()))
        for t in list(self.tfs.keys()):
            self.tfs[t] = self.tfs[t]/max_val
    def calc_weights(self,word_bag):
        wIJ_dict = {}
        n = len(word_bag)
        for tf in list(self.tfs.keys()):
            df_i = word_bag[tf]
            w_ij = self.tfs[tf] * math.log((n/df_i),2)
            wIJ_dict[tf] = w_ij
        self.weights = wIJ_dict
    #def cosine():
        
    #def okapi():


# In[42]:

def get_stopwords(fp,length):
    stop_words = {}
    sf = open(fp + "stopwords-" + length +".txt","r")
    for sw in sf:
        sw = sw.rstrip("\n\r") 
        stop_words[sw] = sw
    sf.close()
    return stop_words
#Returns a word bag containing all unique terms from our document collection
#Also returns our document Vector objects containing normalized term 
#frequencies
def vectorize(path,sw_length,threshold):
    p = source.PorterStemmer()
    folders = os.listdir(path)
    sw_dict = get_stopwords("data/",sw_length)
    translator = str.maketrans('', '', string.punctuation)
    word_bag = {}
    docs = []
    
    for f in folders:
        fold_path =  path + f + "/" 
        tfiles = os.listdir(fold_path)
        #Go through every document
        for tf_name in tfiles:
            tf = open(fold_path + tf_name,"r")
            term_freqs = {}
            for line in tf:
                words = line.split()
                for w in words:
                    #remove trailing newlines and lower case the term
                    w = w.rstrip("\n\r").lower()
                    #remove punctuation
                    w = w.translate(translator)
                    #stem the term
                    w = p.stem(w, 0,len(w)-1)
                    #Add term to word bag and document class freq idx
                    if(w not in sw_dict and len(w) > 0):  
                        if w not in term_freqs:
                            term_freqs[w] = 1
                            #check if term is in global vocab, if not add to word bag
                            if w not in word_bag:
                                word_bag[w] = 1
                            #increment # of doucements that contain this term
                            else:
                                word_bag[w] +=1
                        else:
                            term_freqs[w] += 1           
            tf.close()
            docs.append(Vector(term_freqs,threshold))    
    return [word_bag,docs]
#Goes through every vector object and calculates the TF-IDF weight with the word_bag dict-> contains # of docs
#a term occurs in. Also has the list of vector objects with the tf-ij values.
def w_terms(word_bag,docs):
    for d in docs:
        d.calc_weights(word_bag)


# In[101]:

res = vectorize("data/c50train/","long",6)
word_bag =  res[0]
docs = res[1]
w_terms(word_bag,docs)

