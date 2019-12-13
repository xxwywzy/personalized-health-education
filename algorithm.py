# coding:utf8

import urllib
import sys

from TextRank import ImproveTextRank
from TFIDF import ImporveTFIDF

# ALGORITHM : KEYWORD EXTRACTION
# INPUT: text/url/keywordNum/weight

def KeywordExtraction(inputDict):
    
    keywordResult = ""
    text = inputDict['text']
    if not text: return keywordResult ## end here
        #url = inputDict.get('url',default=None)
        #result = urllib.urlopen(url)
        #text = result.read().strip()
        #if not text:return keywordResult ## end here
    
    keywordResult = ""
    if inputDict['mode'] == 0:
        keywordResult = ImproveTextRank(text,inputDict['keywordNum'],inputDict['weight'])
    else:
        keywordResult = ImporveTFIDF(text,inputDict['keywordNum'],inputDict['weight'])
    
    return keywordResult
    


