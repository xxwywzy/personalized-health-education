# coding:utf-8
from Function import *

def ImporveTFIDF(text,keywordNum,weight,blackwordFile = r"./Dictionary/keywordBlackList.txt"):
    
    #print text.encode("utf-8")
    segResult = SegText(text)
    TF_Value = GetWordPara(segResult).copy()
    IDF_Value = Load_IDFFromFile()
    
    stopWordList = LoadKeywordBlackList(blackwordFile)
    wordDict = {}
    for word in TF_Value.keys():
        if word in stopWordList: continue
        if word in IDF_Value.keys():
            wordDict[word] = TF_Value[word]*IDF_Value[word]
        else:
            wordDict[word] = TF_Value[word]*IDF_Value['DEFAULT']
    
    ## Get KeywordList
    resultList = ArrangeKeyword(wordDict,keywordNum,weight)
    return resultList

def Load_IDFFromFile(filename = r"./Dictionary/idf_5000.txt"):
    
    IDF_Value = {}
    openIDFfile = open(filename,"rb")
    readLines = openIDFfile.readlines()
    stopCode = chardet.detect(readLines[0])['encoding']
    
    for line in readLines:
        ustring = line.decode("utf-8")
        ustring = ustring.strip()
        sline = ustring.split(u" ")
        try:
            word,idf = sline
            IDF_Value[word] = float(idf)
        except:
            continue
    return IDF_Value