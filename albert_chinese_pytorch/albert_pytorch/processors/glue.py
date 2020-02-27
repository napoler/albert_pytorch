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
""" GLUE processors and helpers """
# import Terry_toolkit as tkit
import  tkitFile
import logging
import os
import torch
from .utils import DataProcessor, InputExample, InputFeatures

from random import sample

logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    # all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
        # elif output_mode == "terryner": #回归
        #     label_dict=processor.get_labels_dict()
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        
        #自动屏蔽15%的文字信息
        maskid=tokenizer.convert_tokens_to_ids("[MASK]")
        # print(inputs["input_ids"])
        if len(inputs["input_ids"])>10:
            for num in sample(range(1,len(inputs["input_ids"])),int(0.15*len(inputs["input_ids"]))):
                try:
                    inputs["input_ids"][num] =maskid[0]
                except:
                    pass
                # text_a="".join(text_a)
        
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        # print(label_map)
        # print("label",example.label)
        # #ner需要修改 label应该预先产生就可以了
        # print('example.label',example.label)
        # print('example.label type',type(label_map))
        # for i in label_map.keys():
        #     print(type(i))
        # print('ids len',len(input_ids))
        if output_mode == "classification": #分类
            try:
                label = label_map[example.label]
            except KeyError:
                # print("Error",KeyError)
                label=label_map.get(int(example.label))
                
        elif output_mode == "regression": #回归
            label = float(example.label)
        elif output_mode == "terryner": #回归
            # label_dict=processor.get_labels_dict()
            # label =tokenizer.convert_tokens_to_ids(example.label.split(' ')) + ([pad_token_segment_id] * padding_length)
            label=[]
            # label.append(label_map['[CLS]'])

            for it in  example.label.split(' '):
                label.append(label_map[it])
            if len(label)>=max_length-2:
                label=label[:max_length-2]
            # label.append(label_map['[SEP]'])
            label=[label_map['[CLS]']]+label+[label_map['[SEP]']]+[0] * padding_length
            # print('ids len',len(input_ids))
            # print('len(label)',len(label))
            # print(label)


         
            # print(label)
            # label = float(example.label)
        else:
            raise KeyError(output_mode)
        # print("input_ids",input_ids[0])
        # print("label",label)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            # logger.info("label: %s" % " ".join([str(x) for x in label]))
            # logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          input_len=input_len))
    return features

# class ＴerryProcessor(DataProcessor):
#     """Processor for the MRPC data set (GLUE version)."""

#     def get_train_examples(self, data_dir):
#         """See base class."""
#         logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = line[3]
#             text_b = line[4]
#             label = line[0]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
class TerryNerProcessor(DataProcessor):
    """Processor for 实体识别"""
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")


    def get_labels(self):
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
    def get_labels_dict(self):
        labels={}
        for i,item in enumerate(self.get_labels()):
            labels[item] =i
        return labels
            

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # print(line)
            #   text = tokenization.convert_to_unicode(line[1])
            text_a=line[1]
            label = line[0]
        #     text_b=line[1]
        #    label = 1
            # print(len(text_a))
            # print(len(label))
            #   label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=str(text_a), text_b=None,label=label))
        return examples

    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

class TerrykgProcessor(DataProcessor):
    """Processor for  Terry知识抽取训练　使用百度训练集https://dataset-bj.cdn.bcebos.com/sked/train_data.json"""
    def __init__(self):
        self.data_dir="dataset/terrykg"


    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        self.data_dir=data_dir
        file_path = os.path.join(self.data_dir,"train.json")
        # bulid_labels(self)
        tjosn=tkitFile.Json(file_path=file_path).auto_load()
        return self._create_examples(tjosn, 'train')
    def get_dev_examples(self, data_dir):
        """See base class."""
        self.data_dir=data_dir
        file_path = os.path.join(self.data_dir,"dev.json")
        # bulid_labels(self,data_dir)
        tjosn=tkitFile.Json(file_path=file_path).auto_load()
        return self._create_examples(tjosn, 'dev')

    def bulid_labels(self):
        """See base class　基于数据构建ｌａｂｅｌ词典."""
        # print("self.data_dir",self.data_dir)
        file_path = os.path.join(self.data_dir,"all_50_schemas.json")
        data=tkitFile.Json(file_path=file_path).auto_load()
        # tjson=tkit.Json(file_path=os.path.join('data/all_50_schemas.json'))
        # data= tjson.auto_load()
        # print(len(data))
        labels=[]
        for i,it in enumerate(data):
            labels.append(it['predicate'])
        # labels = list(set(labels))  
        labels = {}.fromkeys(labels).keys()
        # print(labels)
        labels_dict ={"NULL":0}
        for i,it in enumerate(labels,1):
            labels_dict[it]=str(i)
        # print(labels_dict)
        return labels_dict
    def get_labels(self):
        """See base class."""
        labels_dict=self.bulid_labels()
        # for i,it in enumerate(labels_dict):
        # print(range(0,len(labels_dict))
        label=[]
        for i in range(len(labels_dict)):
            label.append(i)
        # print('label',label)

        return label
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        labels_dict=self.bulid_labels()
        examples = []
        for (i, line) in enumerate(lines):
            
            text= line['text']
            for n in line['spo_list']:
                guid = "%s-%s-%s" % (set_type, i,n)
                text_a=text+" [SEP]  "+n['object']+" [SEP]  "+n['subject']
                # print(n['predicate'])
                label = labels_dict[n['predicate']] #这里获取关系对应的编号
                # print("label",label)
                examples.append(
                    InputExample(guid=guid, text_a=str(text_a), text_b=None, label=str(label)))
        # print("语料数量",len(examples))
        return examples
# TerrykgProcessor().get_labels()








class TerryProcessor(DataProcessor):
    """Processor for 自定义数据集 10分类"""
    def __init__(self):
        self.data_dir='dataset/terry_rank'
        # self.make_labels()
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir,"train.json")
        tjosn=tkitFile.Json(file_path=file_path).load()
        return self._create_examples(tjosn, 'train')
    def get_dev_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir,"dev.json")
        tjosn=tkitFile.Json(file_path=file_path).load()
        return self._create_examples(tjosn, 'dev')
    def make_labels(self):
        tjosn=tkitFile.Json(file_path=self.data_dir+"/labels.json").auto_load()
        labels=[]
        for item in tjosn:
            labels.append(str(item['label']))
        self.labels=labels

    def get_labels(self,n=3):
        """See base class.
        设置分类数目
        """
        # return ["0", "1", "2", "3","4","5","6","7","8","9"]
        # return ["0", "1", "2", "3","4","5","6","7","8","9"]
        # print("self.labels",self.labels)
        # return self.labels
        
        return [i for i in range(n)]
 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            # text_a=list(text_a)
            # #自动屏蔽百分之15的数据
            # for num in sample(range(1,len(text_a)),0.15*len(text_a)):
            #     text_a[num] ="[MASK]"
            # text_a="".join(text_a)

            label = line['label']

            # if it['label'] not in [0,1]:
            if label in self.get_labels():
                # print("label",label)
                examples.append(
                    InputExample(guid=guid, text_a=str(text_a), text_b=None, label=str(label)))
                # print("语料数量",len(examples))
            else:
                continue
            
        return examples

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class XnliProcessor(DataProcessor):
    """Processor for the Xlni data set (GLUE version)."""

    def __init__(self):
        self.language = 'zh'

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if label == "contradictory":
                label = "contradiction"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = line[0]
            if language != self.language:
                continue
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "test-%d" % (i)
            language = line[0]
            if language != self.language:
                continue
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class InewsProcessor(DataProcessor):
    """Processor for the inews data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        labels = ["0", "1", "2"]
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = line[3]
            if set_type == "test222":
                label = "0"
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TnewsProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            if set_type == 'test':
                label = '0'
            else:
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class LcqmcProcessor(DataProcessor):
    """Processor for the LCQMC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            if set_type == 'test222':
                label = '0'
            else:
                label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {
    "mnli": 3,
    "mrpc": 2,
    "terry": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "xnli": 3,
    'tnews': 15,
    'lcqmc': 2,
    'inews': 3,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "terry": TerryProcessor,
    "terrykg": TerrykgProcessor,
    "terryner": TerryNerProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    'tnews': TnewsProcessor,
    'xnli': XnliProcessor,
    'lcqmc': LcqmcProcessor,
    'inews': InewsProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "terry": "classification",
    "terrykg": "classification",
    "terryner": "terryner",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    'tnews': "classification",
    'xnli': "classification",
    'lcqmc': "classification",
    'inews': "classification",
}
