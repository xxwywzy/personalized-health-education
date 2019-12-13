# coding:utf8

import networkx as nx
import numpy as np
import chardet

from Function import *

# TextRank 词矩阵运算
def combine(word_list,windows = 5):
    
    if windows < 2: windows = 2
    for x in range(1,windows):
        if x >= len(word_list):break
        word_list2 = word_list[x:]
        res = zip(word_list,word_list2)
        for r in res:
            yield r
            
def ImproveTextRank(text,keywordNum,weight,windows = 5,blackwordFile = r"./Dictionary/keywordBlackList.txt"):
    
    segResult = SegText(text)
    TF_Value = GetWordPara(segResult).copy()
    
    stopWordList = LoadKeywordBlackList(blackwordFile)

    ## filter the characteristic and word
    global endMark,filterNounlist,filterVerblist
    
    vertex_source = []
    wordlist = []

    filterFlag = filterNounlist+filterVerblist
    for word,flag in segResult:
        if word in endMark:
            vertex_source.append(wordlist)
            wordlist = []
        if flag[0] not in filterFlag: continue
        elif word in stopWordList: continue
        wordlist.append(word)
    
    # use the windows create the chain
    # vertex_source is regard as same as the edge? 
    word_index = {}
    index_word = {}
    words_number = 0
    for word_list in vertex_source:
        for word in word_list:
            if word not in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1
    
    graph = np.zeros((words_number,words_number)) # create a TD
    
    for word_list in vertex_source:
        for w1,w2 in combine(word_list, windows):
            if w1 in word_list and w2 in word_list:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index2] = 1.0
                
    nx_graph = nx.from_numpy_matrix(graph) #图计算
    scores = nx.pagerank(nx_graph, **{'alpha': 0.85,}) # set the para is 0.85
    
    wordDict = {}
    for index in scores.keys():
        word = index_word[index]
        if word in TF_Value.keys():
            wordDict[word] = scores[index] * TF_Value[word]

    ## get the keyword result
    resultList = ArrangeKeyword(wordDict,keywordNum,weight)
    return resultList

    
