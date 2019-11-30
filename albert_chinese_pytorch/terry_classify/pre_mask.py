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

from .model.modeling_albert import BertConfig, AlbertForSequenceClassification,AlbertForMaskedLM,AlbertForNextSentencePrediction,AlbertForMultipleChoice,AlbertForTokenClassification,AlbertForQuestionAnswering
from .model.tokenization_bert import BertTokenizer
# from .model.file_utils import WEIGHTS_NAME
# from .model.optimization import AdamW, WarmupLinearSchedule

# from .metrics.glue_compute_metrics import compute_metrics
# from .processors import glue_output_modes as output_modes

# from .processors import glue_processors as processors
# from .processors import glue_convert_examples_to_features as convert_examples_to_features
# from .processors import collate_fn

# from .tools.common import seed_everything
# from tools.common import init_logger, logger
# from callback.progressbar import ProgressBar

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())
#model/modeling_albert.py 文件内
MODEL_CLASSES = {
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer),
    'albert_mlm': (BertConfig, AlbertForMaskedLM, BertTokenizer), #mlm
    'albert_sentence_pre': (BertConfig, AlbertForNextSentencePrediction, BertTokenizer),
    'albert_mul_choice': (BertConfig, AlbertForMultipleChoice, BertTokenizer),
    'albert_token_class': (BertConfig, AlbertForTokenClassification, BertTokenizer),
    'albert_qa': (BertConfig, AlbertForQuestionAnswering, BertTokenizer),
}

class Mask:
    def __init__(self, model_name_or_path='prev_trained_model/albert_tiny',max_length=512):
        """
        使用模型进行分类操作
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device",self.device)
        config_class, model_class, tokenizer_class = MODEL_CLASSES['albert_mlm']
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path,
                                                    do_lower_case=False)
        config = config_class.from_pretrained(model_name_or_path,num_labels=5,finetuning_task='mlm',output_attention=True,share_type='all')
        self.model = model_class.from_pretrained(self.model_name_or_path,config=config)
    def pre(self,text):
        """
        这里进行预测结果
        >>>pre(text)
        """
        # text=text[:self.max_length]
        # print(text)

        # inputs = self.tokenizer.encode_plus(text,'',   add_special_tokens=True, max_length=self.max_length)
        # print(inputs)
        # # token_type_ids=segments_tensors

        # # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        # # input_ids = torch.tensor(input_ids)  # Batch size 1
        # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # token_type_ids=

        # # print("input_ids",input_ids)
        # # print(inputs)


        text = "[CLS]天气好吗 [SEP] 是吗 [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_index = 5
        tokenized_text[masked_index] = '[MASK]'
        # assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

        print(tokenized_text)
        print(len(tokenized_text))
        # Convert token to vocabulary indices
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0, 0, 0, 0, 0,0, 1, 1, 1]
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
        # input_ids = torch.tensor(self.tokenizer.encode("你 好 吗 天 起 很 好")).unsqueeze(0)  # Batch size 1
        token_type_ids = torch.tensor(segments_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
        # outputs = self.model(input_ids, token_type_ids=token_type_ids)
        outputs = self.model(input_ids, masked_lm_labels=input_ids,token_type_ids=token_type_ids)
        self.model.to(self.device)
        # print(outputs)
        loss, prediction_scores = outputs[:2]
        # print(loss)
        # print(prediction_scores)
        predicted_index = torch.argmax(prediction_scores[0, 1]).item()
        print(predicted_index)
        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token)
        # for mask_index in range(len( segments_ids)):
        #     print("mask_index",masked_index)
        #     # if mask_index>2:
        #     #     predicted_index = torch.argmax(prediction_scores[mask_index-2, mask_index]).item()
        #     # else:
        #     #     predicted_index = torch.argmax(prediction_scores[0, mask_index]).item()
        #     predicted_index = torch.argmax(prediction_scores[0, mask_index]).item()
        #     print(predicted_index)
        #     predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        #     print(predicted_token)
        # self.model.save_pretrained('model/t')
        # self.tokenizer.save_pretrained('model/t')

    
if __name__ == "__main__":
    text="[CLS]天气真好啊 [SEP] 是吗 [SEP]"
    tclass=Mask(model_name_or_path='prev_trained_model/albert_tiny') 
    print("预测结果",tclass.pre(text))
