# coding:utf-8

from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import jieba.analyse
import os
from algorithm import KeywordExtraction
import pandas as pd
import ast

# 读取模型
global model1
model1 = Word2Vec.load('../models/1112_1554_text.model')

# 标签向量初始化
def initial_vec():
    vec = {
    "男性": 0,
    "女性": 0,
    "老年": 0,
    "青年": 0,
    "怀孕": 0,
    "肥胖": 0,
    "心理": 0,   
    "吸烟": 0,
    "饮酒": 0,
    "饮食": 0,
    "运动": 0,
    "血压": 0,
    "血糖": 0,
    "胆固醇": 0,
    "甘油三酯": 0,
    "脂蛋白": 0,
    "血尿酸": 0,
    "高血压": 0,
    "糖尿病": 0,
    "慢阻肺": 0,
    "脑卒中": 0,
    "冠心病": 0,
    "高血脂": 0,
    "皮肤病": 0,
    "肾病": 0,
    "眼病": 0,
    "肺病": 0,
    "肝病": 0,
    "胃病": 0,
    "降压药": 0,
    "降糖药": 0,
    "胰岛素": 0,
    "降脂药": 0,
    }
    return vec

# 预处理关键词列表
def preprocess_keyword_list(keyword_list):
    new_keyword_list = [] #新建一个list，对原list操作会跳过部分item
    for keyword in keyword_list:
        if keyword not in model1.wv: #不存在的组合词先分一次词再处理，再不在就不管了
            jieba.del_word(keyword) #删除该词，进行进一步切分
            for atom in jieba.cut(keyword):
                if atom in model1.wv:
                    new_keyword_list.append(atom)
        else:
            new_keyword_list.append(keyword)
    return new_keyword_list

# 计算文本向量
def calculate_text_vec(keyword_list):
    text_vec = initial_vec()
    preprocesed_keyword_list = preprocess_keyword_list(keyword_list)
    print("已处理的关键词：", preprocesed_keyword_list)
    for keyword in preprocesed_keyword_list:
        if keyword in text_vec.keys():
            print("%s 位于标签中，已完成映射" % keyword)
            text_vec[keyword] = 1
        else:
            similarity_dic = {}
            for tag in text_vec.keys():
                similarity_dic[tag] = model1.wv.similarity(tag,keyword) 
            sort_list = sorted(similarity_dic.items(), key=lambda d: d[1], reverse=True)
            for item in sort_list:
                if(item[1] >= 0.3):
                    text_vec[item[0]] += item[1] #权重叠加
                    print(keyword, "权重大于 0.3 的标签", item)
    return text_vec

# 得出该关键词列表与本体标签相关的标签列表（暂时不用）
def calculate_tag_list(keyword_list):
    tag_list = []
    text_vec = initial_vec()
    preprocesed_keyword_list = preprocess_keyword_list(keyword_list)
    for keyword in preprocesed_keyword_list:
        if keyword in text_vec.keys():
            tag_list.append(keyword)
        else:
            similarity_dic = {}
            for tag in text_vec.keys():
                similarity_dic[tag] = model1.wv.similarity(tag,keyword) 
            sort_list = sorted(similarity_dic.items(), key=lambda d: d[1], reverse=True)
            for item in sort_list:
                if(item[1] >= 0.3):
                    tag_list.append(item[0])
    return list(set(tag_list)) #去除重复

# 计算患者向量与文本向量的内积
def calculate_inner_product(patient_vec, text_vec):
    patient_array = np.array(list(patient_vec.values()))
    text_array = np.array(list(text_vec.values()))
    result = np.dot(patient_array, text_array)
    return result

# 计算患者向量与文本向量的内积
def calculate_inner_product_new(patient_vec, text_vec):
    patient_array = np.array(patient_vec)
    text_array = np.array(list(text_vec.values()))
    result = np.dot(patient_array, text_array)
    return result


def rename(path): #注意命名过一次之后不要重复命名，会替换原来的文件，导致数量减少
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.txt'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), '0000' + format(str(i), '0>3s') + '.txt') #命名为test+三位数
                os.rename(src, dst)
                i = i + 1
        print('total %d to rename & converted %d txts' % (total_num, i))

# 预处理文本(一篇)
def preprocess_text(file_path):
    with open(file_path,'r') as f:
        str_list = []
        index = 1
        for line in f:
            if (index != 1): #标题单独一行，正文合并
                line = line.strip('\n')
            str_list.append(line)
            index += 1
        text_str = ''.join(str_list)
        return text_str

# 生成inputDict
def generate_input(text,mode=0,weight=0,keywordNum=5):
    inputDict = {}
    inputDict["text"] = text
    inputDict["mode"] = mode #textrank
    inputDict["weight"] = weight
    inputDict["keywordNum"] = keywordNum
    return inputDict

# 处理文本库
def preprocess_corpus(path):
    filelist = os.listdir(path)
    improved_textrank_total_list = []
    improved_tfidf_total_list = []
    original_tfidf_total_list = []
    original_textrank_total_list = []
    item_list = []
    filelist.sort()
    for item in filelist:
        if item.endswith('txt'):
            src = os.path.join(os.path.abspath(path), item)
            text_str = preprocess_text(src)
            tr = jieba.analyse.TextRank()
            tr.span = 2
            original_textrank_list = list(tr.textrank(text_str, topK=5)) #停用词词典和idf词典使用默认，不共享（原始tfidf效果很差）
            original_tfidf_list = list(jieba.analyse.extract_tags(text_str, topK=5)) # 先执行原始的算法，防止之后的自定义词典的影响 
            original_textrank_total_list.append(original_textrank_list)
            original_tfidf_total_list.append(original_tfidf_list)
            inputDict_textrank = generate_input(text_str,mode=0)
            inputDict_tfidf = generate_input(text_str,mode=1)
            improved_textrank_list = KeywordExtraction(inputDict_textrank) # 会把组合词放到词典
            improved_tfidf_list = KeywordExtraction(inputDict_tfidf) # 会把组合词放到词典
            improved_textrank_total_list.append(improved_textrank_list)
            improved_tfidf_total_list.append(improved_tfidf_list)
            item_list.append(item)
            # tag_list = calculate_tag_list(keyword_list)
            print(item, "关键词：", improved_textrank_list)
    # print(len(item_list),len(keyword_total_list))
    df = pd.DataFrame({'text': item_list, 'improved_textrank': improved_textrank_total_list,'original_textrank': original_textrank_total_list, 
        'improved_tfidf': improved_tfidf_total_list,'original_tfidf': original_tfidf_total_list})
    df.to_csv("test_text_100_newest.csv",index=False)


def generate_product_results(text_path, patient_path, method):
    df_text = pd.read_csv(text_path)
    df_patient = pd.read_csv(patient_path)
#     df_patient.pop('患者描述')
    patient_list = df_patient.values.tolist()
    for item in patient_list:
        product_list = []
        index_list = []
        patient_vec = item[1:]
        patient_no = item[0]
        for index,row in df_text.iterrows():
            keyword_list = ast.literal_eval(row[method])
            text_vec = calculate_text_vec(keyword_list)
            product = calculate_inner_product_new(patient_vec,text_vec) #内积的精度有限
            product_list.append(product)
            index_list.append(index)
        df_text['文本编号'] = index_list
        df_text['患者'+str(patient_no)] = product_list
    df_text.pop('text')
    df_text.pop('original_tfidf')
    df_text.pop('original_textrank')
    df_text.pop('improved_tfidf')
    df_text.pop('improved_textrank')
    df_text.pop('manual')
    df_text.to_csv("patient_text_result_"+ method + ".csv",index=False)

    



