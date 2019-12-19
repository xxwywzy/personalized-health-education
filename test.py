# coding:utf-8

from algorithm import KeywordExtraction
import Vector
import jieba.analyse

# 处理文本库，生成关键词列表
def generate_corpus():
	text_path = '/Users/wangzheyu/test'
	Vector.preprocess_corpus(text_path)

# 处理患者向量和关键词，生成对应的内积
def generate_product():
	text_path = './data/window-2-result/test_text_100.csv'
	patient_path = './data/patient_vec_50.csv'
	Vector.generate_product_results(text_path, patient_path, 'manual')

if __name__ == '__main__':
	generate_product()


# for x, w in jieba.analyse.textrank(s, topK=5,withWeight=True):
#     print('%s %s\n' % (x, w))
    

# inputDict 代替POST or GET 的输入数据
# inputDict = {}
# inputDict["text"] = s
# inputDict["mode"] = 0 #textrank
# inputDict["weight"] = 0
# inputDict["keywordNum"] = 5
# keyword_list = KeywordExtraction(inputDict) # 会把组合词放到词典
# print("提取关键词：", keyword_list)

# # 调用 algorithm.py 中的KeywordExtraction函数，按传入参数形式选择功能即可调用

# patient_vec = Vector.generate_patient_vector()
# text_vec = Vector. calculate_text_vec(keyword_list)
# print(Vector.calculate_inner_product(patient_vec,text_vec))
