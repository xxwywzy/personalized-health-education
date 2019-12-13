#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
# import logging
import os
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
def trainWord2vecModel():
    
    # program = os.path.basename(sys.argv[0])
    # logger = logging.getLogger(program)
 
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(sys.argv))  # logging 输出进度
 
    # check and process input arguments
    inp = u"./models/1112_1554_model_no_blackword.zh.text"
    outp1 = u"./models/1112_1554_text.model"
    outp2 = u"./models/1112_1554_text.vector"
 
    model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
 
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
