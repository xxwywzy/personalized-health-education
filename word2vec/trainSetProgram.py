# -*- coding: utf-8 -*-

import os,chardet
import jieba
import jieba.posseg as pseg

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# src_path 待分词处理的训练文本所在文件夹
# blackwordlist_path 主题词停用黑名单文件
#==建议输入的文件路径为unicode编码 如 src_path = u"xxx/yyy"
def getTrainingSetTexts(src_path,output_file_path=u"./models/1112_1554_model_no_blackword.zh.text",blackwordlist_path = u"./dictionary/keywordBlackList.txt"):

    stopWord = getWordListFromFile(blackwordlist_path) #加载停用词典
    stopFlag = [u'x',u'e',u'm'] # 停用词性包括：x:符号、e:英文、m:数字
    
    output_file = open(output_file_path,"w")

    # 遍历文件夹下的文件，将训练用的文本放在单个文件夹下遍历
    if not os.path.exists(src_path):
        print "没有这个训练集文件夹"

    jieba.load_userdict(u"./dictionary/userdict.txt")
    walk = os.walk(src_path)
    fileCount = 0
    for root, dirs, files in walk:
        # 获取文本路径
        allCount = len(files)
        for fn in files:
            
            f = open(os.path.join(root,fn),'r')
            raw = f.read()
            f.close()
            
            # 获得分词结果后过滤停用词、无用词性
            try:
                result = pseg.cut(raw)
                if result: 
                    word_list=[]
                    for word,flag in result:
                        if flag[0] in stopFlag: continue
                        if word in stopWord: continue
                        word_list.append(word)
                    output_file.write(u" ".join(word_list) + u"\n")
                    fileCount += 1
                    if fileCount%500 == 0 : print "进度%d/%d" % (fileCount,allCount)
            except Exception:
                print e
    
    output_file.close()
    print "处理完成"

# 从文件中读取词语列表，存于数组中返回
# 文件中的数据格式： 每词一行 (适用该格式的有：主题词停用表、关键词黑名单等)
def getWordListFromFile(file_path):
    
    if not os.path.exists(file_path): return []

    # 读取文件数据
    stopFile = open(file_path,"r")
    List = stopFile.read()
    stopFile.close()

    # 对文件数据进行编码检测，并转为unicode编码
    tmpList = List
    try:
        cc = chardet.detect(List)['encoding']
        try: List = unicode(List,cc)
        except Exception,e:  List = tmpList
    except Exception,e: pass
    
    # 将文件数据以回车分隔，读出存于WordList列表中
    WordList = []
    stopwordList = List.split(u"\n")
    for stopword in stopwordList:
        if stopword not in WordList:
            WordList.append(stopword)
            
    return WordList