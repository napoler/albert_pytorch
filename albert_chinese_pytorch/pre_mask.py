# coding=utf-8
# 测试分类效国

from terry_classify import Mask
# if __name__ == "__main__":
#     text="[CLS]天气真好啊 [SEP] 是吗 [SEP]"
#     tclass=Mask(model_name_or_path='./prev_trained_model/albert_tiny') 
#     print("预测结果",tclass.pre(text))
import torch
from transformers import *
model = BertForSequenceClassification.from_pretrained('prev_trained_model/albert_tiny')
tokenizer = BertTokenizer.from_pretrained('prev_trained_model/albert_tiny')


tokenizer.add_tokens(['你好吗', '好啊'])
model.resize_token_embeddings(len(tokenizer))
# Train our model
train(model)