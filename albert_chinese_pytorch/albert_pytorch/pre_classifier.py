# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on chineseGLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .model.modeling_albert import BertConfig, AlbertForSequenceClassification
from .model.tokenization_bert import BertTokenizer
from.utils import *
from .plus import *


class classify:
    def __init__(self, model_name_or_path='outputs/terry_output',finetuning_task='finetuning_task',max_length=512):
        """
        使用模型进行分类操作
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device",self.device)
        #自动加载模型和词典
        P=Plus()
        P.args['class_name']="AlbertForSequenceClassification"
        P.args['model_name_or_path']=model_name_or_path
        P.args['finetuning_task']=finetuning_task
        self.model,self.tokenizer,self.config_class=P.load_model()
    def pre(self,text):
        """
        这里进行预测结果
        >>>pre(text)
        """
        input_ids, token_type_ids=Plus().encode(text=text,tokenizer=self.tokenizer,max_length=self.max_length)
        outputs = self.model(input_ids)
        self.model.to(self.device)
        # print(outputs)
        seq_relationship_scores = outputs[0] #对应的概率信息
        # print(seq_relationship_scores)
        # print( torch.argmax(seq_relationship_scores).item())
        return torch.argmax(seq_relationship_scores).item()
    
if __name__ == "__main__":
    text="养龟之前要想好的问题\n想养龟之前先问自己能否承受寂寞。恐怕没有人会想到十年后，二十年，甚至五十年后的今天这条龟降临何方。龟龟虽小，养起来真是一个足够大的包袱，没有充分的思想准备。理解龟龟的生长其实养龟最辛苦的时期是小龟饲养后的第二年，第一年一般兴趣高涨，对于龟龟的照顾无微不至，看着龟龟生长迅速，看着龟龟天天长大，心里的满足溢于言表，但养了几年后，龟龟的生长开始变缓，体长每月1公分甚至不到的变化，经常使龟友开始耐不住寂寞，开始埋怨生长的缓慢，其时兴趣与耐心正在一点点失去。但在这个时期，自己的爱龟刚刚成年，正在向成熟进发，爱龟的一天天的熟悉自己，幼稚的身躯慢慢挂上凝重的颜色，如此的过程不正是养龟人期待的过程，不正是成功的体验？不要被别人的意见所左右很多朋友将自己的龟贴出来供大家赏评，赞美的话是最喜欢听的（无论谁，除非是受虐狂），但龟友很多意见有好自然也有不好，要有心里承受能力，否则，最好不贴，别人一说这是烂甲，这品相不好，这是隆背，马上对爱龟的热爱从顶峰掉到谷底，回家怎么看自己的龟龟怎么不顺眼，完了，不久的将来，这条龟一定不存在于这个窝。养龟不是为别人养的，自己的龟只属于自己，让别人说去吧，能把一条巴西养十年以上值得佩服。喜新厌旧要不得交流的时间多了，自己的鉴赏力提高，也见过很多好龟，于是缅陆龟养一段时间换印星，印星换小苏，小苏换靴脚，又养缅星，还有更好的辐射，喜新厌旧使自己变成了一个快乐的饲养员，不停的饲养着，没有一刻的宁静，也慢慢消耗了自己的耐心，因为没有人喜欢终日耕种没有收获，养龟的乐趣是在漫长龟龟成熟的过程中体验。图片来源于网络"
    tclass=classify(model_name_or_path='../outputs/terry_output') 
    print("预测结果",tclass.pre(text))
