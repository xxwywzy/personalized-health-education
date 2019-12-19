# coding:utf-8

import jieba,chardet
import jieba.posseg as pseg
import math
import os

# ================== Function ================

global endMark,compoundWordFlag,filterNounlist,filterVerblist # 也许可以不用声明（默认全局）
endMark = [u'。',u'.',u'，',u',',u'\n']
compoundWordFlag = [u'n',u'v',u'a',u'b',u'j',u'l',u'f'] # 添加方位词词性
filterNounlist = [u'n',u'l',u'j',u't']
filterVerblist = [u'v',u'f']
blackFlag = [u'ns']

# ================= TEXT SEG ==============
## change the ltrator to list wit tuple
def SegText(text):
    
    # problem 1: text is not a unicode object?
    ## getSegResult with compound word, turn to list
    result = pseg.cut(text)
    textList = []
    for word,flag in result:
        textList.append((word,flag))
    
    ## get compoundUserdict
    compoundUserdict = CompoundWord(textList)
    jieba.load_userdict(compoundUserdict) #添加组合词后会对分词判断产生影响
    ## seg text 
    result = pseg.cut(text)
    segTextList = []
    for word,flag in result:
        segTextList.append((word,flag))

    return segTextList

# ================== GET KEYWORD ===============
def ArrangeKeyword(wordDict,keywordNum,weight):
    
    dict = sorted(wordDict.items(),key = lambda item:item[1],reverse = True)
    
    resultList = []
    for word,score in dict:     
        flag = 0
        for old_word in resultList:
            if CosWord(word,old_word):
                flag = 1
                if len(word) >= len(old_word):
                    i = resultList.index(old_word)
                    resultList[i] = word # 可能会导致权重稍低的词语排序靠前
            # resultList = list(set(resultList)) # get the intersection（取不重复的部分，会改变词语顺序，似乎没有用）
        if flag: continue
        resultList.append(word)
        if len(resultList) >= keywordNum:break
    
    # outputResult = ""
    # for keyword in resultList:
    #     if not weight:
    #         outputResult += "%s\n" % keyword
    #     else:
    #         outputResult += "%s %.10f\n" % (keyword,wordDict[keyword])
    # outputResult = outputResult[:-1] #除去最后一个换行符
    
    return resultList
    
def LoadKeywordBlackList(blackwordFilePath):
    
    stopWordList = []

    stopFile = open(blackwordFilePath,"rb")
    List = stopFile.read()
    code = chardet.detect(List)['encoding']  # return the coding UTF8-SIG
    unicodeList = str(List,code)
    wordList = unicodeList.split(u"\r\n")

    for stopword in wordList:
        if stopword not in stopWordList:
            stopWordList.append(stopword)
            
    return stopWordList

def GetWordPara(segTextList,titleValue = 3.00,nounValue = 1.200,verbValue = 0.800):
    
    TF_Dict = {}
    # Now title in the content
    global compoundWordFlag,filterNounlist,filterVerblist

    cc = True # First line is title
    value_Dict = {}
    for word,flag in segTextList:

        if cc:
            if word == u"\n": cc = False
            else:
                value_Dict[word] = titleValue
                
        if flag[0] in filterNounlist: 
            if word in value_Dict.keys():
                if value_Dict[word] < nounValue:
                    value_Dict[word] = nounValue
            else:
                value_Dict[word] = nounValue

        elif flag[0] in filterVerblist: 
            if word not in value_Dict.keys():
                value_Dict[word]=float(1.0)
        else:
            continue
        
        if word not in TF_Dict.keys():TF_Dict[word] = 1
        else:TF_Dict[word] += 1
                
    ## calculate TF_Value
    for word in TF_Dict.keys():
        value = TF_Dict[word]
        TF_Dict[word] = float(value)/(value+1)
    
    wordValue = {}
    for word in TF_Dict.keys():
        wordValue[word] = TF_Dict[word] * value_Dict[word]

    return wordValue
    
def CompoundWord(segTextList,compoundT=3):
    
    # set the picture of diagraph by form of dictionary
    # 移除停用词列表，使用词性判断
    global compoundWordFlag
    
    length = len(segTextList)
    wordDigraph = {}
    counter = 0-1
    while 1:     
        counter += 1
        if counter >= length: break
               
        word,flag = segTextList[counter]
        if flag[0] not in compoundWordFlag: continue #只有下一个词符合条件才会开始计算组合词
        
        ## 统计组合词出现的次数
        tmp_compoundword = [word]
        while 1:
            counter += 1
            if counter >= length: break
            
            next_word,next_flag = segTextList[counter]
            if next_flag[0] not in compoundWordFlag: break
            
            tmp_compoundword.append(next_word)
            
        if len(tmp_compoundword) >= 2:
            com = ""
            for word in tmp_compoundword: com += word
            
            if com not in wordDigraph.keys(): wordDigraph[com] = 1
            else: wordDigraph[com] += 1        
    output = ""
    for word in wordDigraph.keys():
        if wordDigraph[word] < compoundT: 
            continue
        else:
            output += "%s n\n" % word
            
    output += u"%s n\n" % word
    output = output[:-1]
    
    tmp_userdict = r"./Dictionary/tmp_compoundword_Dictionary.txt"
    openDictFile = open(tmp_userdict,"wb")
    openDictFile.write(output.encode('utf8'))
    openDictFile.close()
    
    return tmp_userdict

# 计算词语余弦相似度
def CosWord(word1,word2,T=0.7):
    
    wordlist = []
    for i in range(0,len(word1)):
        if word1[i] not in wordlist:
            wordlist.append(word1[i])
    for i in range(0,len(word2)):
        if word2[i] not in wordlist:
            wordlist.append(word2[i])
            
    _vector_list = [[0]*len(wordlist),[0]*len(wordlist)]
    for i in range(0,len(word1)):
        index = wordlist.index(word1[i])
        _vector_list[0][index] = 1
    for i in range(0,len(word2)):
        index = wordlist.index(word2[i])
        _vector_list[1][index] = 1
        
    numerator = 0
    denominator = [0,0]
    for i in range(0,len(wordlist)):
        numerator += _vector_list[0][i] * _vector_list[1][i]
        denominator[0] += _vector_list[0][i] ** 2
        denominator[1] += _vector_list[1][i] ** 2
    denominator[0] = math.sqrt(float(denominator[0]))
    denominator[1] = math.sqrt(float(denominator[1]))
        
    result = float(numerator)/(denominator[0] * denominator[1])
    
    if result>T:return True
    else:return False
