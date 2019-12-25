from __future__ import absolute_import, division, print_function
from .model.modeling_albert import *
from .model.tokenization_bert import BertTokenizer

from .pre_classifier import *
from .pre_mask import *
from .model import *
from .processors import *
from .metrics import *
from .tools import *
from .callback import *
from .utils import *
from .plus import *



import argparse
import glob
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# from albert_pytorch  import BertConfig, AlbertForSequenceClassification
# from albert_pytorch import BertTokenizer
# from albert_pytorch import WEIGHTS_NAME
# from albert_pytorch  import AdamW, WarmupLinearSchedule

# from albert_pytorch  import compute_metrics
# from albert_pytorch  import glue_output_modes as output_modes

# from albert_pytorch  import glue_processors as processors
# from albert_pytorch import glue_convert_examples_to_features as convert_examples_to_features
# from albert_pytorch import collate_fn

# from albert_pytorch  import seed_everything
# from albert_pytorch import init_logger, logger
# from albert_pytorch import ProgressBar
import time


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())
# print(ALL_MODELS)
MODEL_CLASSES = {
    'AlbertForSequenceClassification': (BertConfig, AlbertForSequenceClassification, BertTokenizer),
    'AlbertForMaskedLM': (BertConfig, AlbertForMaskedLM, BertTokenizer),
    'AlbertModel': (BertConfig, AlbertModel, BertTokenizer),
    'AlbertForNextSentencePrediction': (BertConfig, AlbertForNextSentencePrediction, BertTokenizer),
    'AlbertForMultipleChoice': (BertConfig, AlbertForMultipleChoice, BertTokenizer),
    'AlbertForTokenClassification': (BertConfig, AlbertForTokenClassification, BertTokenizer),
    'AlbertForQuestionAnswering': (BertConfig, AlbertForMultipleChoice, BertTokenizer)
}




class Plus:
    """
    各种快速函数
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args={
            'adam_epsilon':1e-8,
            'learning_rate':5e-5,
            'warmup_proportion':0.1,
            'per_gpu_train_batch_size':100,
            'train_batch_size':100,
            'max_steps':-1,
            'class_name':'AlbertForSequenceClassification',
            'fp16':False,
            'model_type':'albert',
            'fp16_opt_level':"O1",
            'max_grad_norm':1.0,
            'local_rank':-1,
            'device':self.device,
            'seed':42,
            'num_train_epochs':10,
            'n_gpu':0,
            't_total':10,
            'weight_decay':0,
            'gradient_accumulation_steps':1,
            'model_type':'albert',
            'data_dir':'',
            'model_name_or_path':'',
            'max_seq_length':512,
            'data_dir': 'dataset/terry',
            'finetuning_task':'terry_test',
            'share_type':'all',
            'num_labels':3,
            'output_dir':"outputs/test"   }
        # model_path=''  
        # self.tokenizer = BertTokenizer.from_pretrained(model_path,   do_lower_case=False)
        pass
    def load_model(self):
        """
        精简加载模型流程
        class_name     'AlbertForSequenceClassification'
                                    'AlbertForMaskedLM'
                                    'AlbertModel'
                                    'AlbertForNextSentencePrediction'
                                    'AlbertForMultipleChoice'
                                    'AlbertForTokenClassification'
                                    'AlbertForQuestionAnswering'
        model_path 模型储存的路径   
        """
        # model_name_or_path= model_path
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args['class_name']]

        config = config_class.from_pretrained(self.args['model_name_or_path'],
                                            num_labels=self.args['num_labels'],
                                            finetuning_task= self.args['finetuning_task'],
                                            share_type=self.args['share_type'])
        # print("config",config)
        tokenizer = tokenizer_class.from_pretrained(self.args['model_name_or_path'],
                                                    do_lower_case=False)
        model = model_class.from_pretrained(self.args['model_name_or_path'], config=config)
        model =model.to(self.device)
        return model,tokenizer,config_class


    def encode(self,text,tokenizer,max_length=512):
        """
        tokenizer 字典
        输入文字自动转换成tensor 并且使用自动尝试使用gpu
        input_ids, token_type_ids
        """
        inputs = tokenizer.encode_plus(text,'',   add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
        token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
        # if torch.cuda.is_available():
        input_ids=input_ids.to(self.device)
        token_type_ids=token_type_ids.to(self.device)
        return input_ids, token_type_ids


    def encode_one(self,text,tokenizer,max_length=512):
        """
        tokenizer 字典
        输入文字自动转换成tensor 并且使用自动尝试使用gpu
        input_ids, token_type_ids
        """
        inputs = tokenizer.encode_plus(text,'',   add_special_tokens=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
        input_ids=input_ids.to(self.device)
        return input_ids

    # def mask(self,text,tokenizer):
    #     # text=""
    #     text_b="柯基犬很牛逼啊"
    #     # inputs = tokenizer.get_special_tokens_mask(text,'',   add_special_tokens=True, max_length=max_length)
    #     inputs = tokenizer.encode_plus(text,text_b,   add_special_tokens=True, max_length=512)
    #     inputs_mask = tokenizer.get_special_tokens_mask(inputs["input_ids"])
    #     input_ids = inputs["input_ids"]
    #     # input_ids = inputs["input_ids"]
    #     print(inputs)
    #     print(inputs_mask)
    #     input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1  # Batch size 1
    #     input_ids=input_ids.to(self.device)
    #     return input_ids

        # [MASK]
        # pass



    def load_data(self ,task, tokenizer, data_type='train'):
        """
        处理加载数据
         返回 
         TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        """
        args=self.args
        if args['local_rank'] not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = glue_processors[task]()
        output_mode = glue_output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args['data_dir'], 'cached_{}_{}_{}_{}'.format(
            data_type,
            list(filter(None, args['model_name_or_path'].split('/'))).pop(),
            str(args['max_seq_length']),
            str(task)))
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args['data_dir'])
            label_list = processor.get_labels()
            if task in ['mnli', 'mnli-mm'] and 'roberta' in args['model_type']:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]

            if data_type == 'train':
                examples = processor.get_train_examples(args['data_dir'])
            elif data_type == 'dev':
                examples = processor.get_dev_examples(args['data_dir'])
            else:
                examples = processor.get_test_examples(args['data_dir'])

            features = glue_convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_length=args['max_seq_length'],
                                                    output_mode=output_mode,
                                                    pad_on_left=bool('xlnet' in args['model_type']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if 'xlnet' in args['model_type'] else 0,
                                                    )
            if args['local_rank'] in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args['local_rank'] == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # print('features',features[0])
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        # print(all_token_type_ids)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        # elif output_mode == "terryner": #回归
        #     # all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        #     all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        # print(all_labels)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
        return dataset

    def train(self,train_dataset, model, tokenizer):
        """ Train the model """
        args=self.args

        args['per_gpu_train_batch_size'] = self.args['per_gpu_train_batch_size'] * max(1, self.args['n_gpu'])
        train_sampler = RandomSampler(train_dataset) if args['local_rank'] == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'],
                                      collate_fn=collate_fn)

        if args['max_steps'] > 0:
            t_total = args['max_steps']
            args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
        else:
            t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
        print('args.t_total',args['t_total'])
        warmup_steps = int(args['t_total'] *args['warmup_proportion'])
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=args['t_total'])
        if args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args['local_rank'] != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank'] ],
                                                            output_device=args['local_rank'] ,
                                                            find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_dataset))
        # logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        #             args.train_batch_size * args.gradient_accumulation_steps * (
        #                 torch.distributed.get_world_size() if local_rank != -1 else 1))
        # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)
    
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        seed_everything(args['seed'])  # Added here for reproductibility (even between python 2 and 3)
        for epoch in range(int(args['num_train_epochs'])):
            # print("111")
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training epoch:'+str(epoch+1))

            for step, batch in enumerate(train_dataloader):
                # print(global_step)
                # print(batch)

                
                model.train()
                batch = tuple(t.to(self.args['device']) for t in batch)
                with torch.no_grad():
                    if self.args['class_name']=='AlbertForSequenceClassification':
                        inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'labels': batch[3]}
                        if args['model_type'] != 'distilbert':
                            inputs['token_type_ids'] = batch[2] if args['model_type']  in ['bert', 'xlnet', 'albert',
                                                                                    'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                    elif self.args['class_name']=='AlbertForMaskedLM':  #lm
                        inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'masked_lm_labels':batch[0],
                                # 'labels': batch[3]
                                }
                        if args['model_type'] != 'distilbert':
                            inputs['token_type_ids'] = batch[2] if args['model_type']  in ['bert', 'xlnet', 'albert',
                                                                                    'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                    elif self.args['class_name']=='AlbertForNextSentencePrediction': #下一句
                        inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                # 'masked_lm_labels':batch[0],
                                'next_sentence_label': batch[3] 
                                }
                        if args['model_type'] != 'distilbert':
                            inputs['token_type_ids'] = batch[2] if args['model_type']  in ['bert', 'xlnet', 'albert',
                                                                                    'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids


                                                                                    
                    elif self.args['class_name']=='AlbertForMultipleChoice': #多分类
                        inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'masked_lm_labels':batch[0],
                                'labels': batch[3]
                                }
                        if args['model_type'] != 'distilbert':
                            inputs['token_type_ids'] = batch[2] if args['model_type']  in ['bert', 'xlnet', 'albert',
                                                                                    'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                    elif self.args['class_name']=='AlbertForTokenClassification': #做ner
                        inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'masked_lm_labels':batch[0],
                                # 'labels': batch[3]
                                }
                        if args['model_type'] != 'distilbert':
                            inputs['token_type_ids'] = batch[2] if args['model_type']  in ['bert', 'xlnet', 'albert',
                                                                                    'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                # #分类
                # outputs = model(**inputs)
                # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                #lm
                # outputs = model(input_ids= batch[0], masked_lm_labels= batch[0])
                outputs = model(**inputs)   
                loss, prediction_scores = outputs[:2]

                # outputs = model(input_ids, masked_lm_labels=input_ids)
                # loss, prediction_scores = outputs[:2]

                print(loss.item())

                if args['n_gpu'] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                if args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                # 验证保存
                # if local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     #Log metrics
                #     if local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args, model, tokenizer)

                # if local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            
            
            # Save model checkpoint 保存模型
            # output_dir=
            # output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
            output_dir = os.path.join(args['output_dir'], 'checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(vocab_path=output_dir)
            pbar(step, {'loss': loss.item()})
            print(" ")
            # if 'cuda' in str(args.device):
            if args['device']=='cuda':
                torch.cuda.empty_cache()
        return global_step, tr_loss / global_step




    def train_ner(self,train_dataloader, model, tokenizer):
        """ Train the model """
        args=self.args
        warmup_steps = int(args['t_total'] *args['warmup_proportion'])
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=args['t_total'])
        if args['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args['local_rank'] != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank'] ],
                                                            output_device=args['local_rank'] ,
                                                            find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_dataset))
        # logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        #             args.train_batch_size * args.gradient_accumulation_steps * (
        #                 torch.distributed.get_world_size() if local_rank != -1 else 1))
        # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)
    
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        seed_everything(args['seed'])  # Added here for reproductibility (even between python 2 and 3)
        for epoch in range(int(args['num_train_epochs'])):
            # print("111")
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training epoch:'+str(epoch+1))

            for step, batch in enumerate(train_dataloader):
                # print(global_step)
                # print(batch)
                model.train()

                input_ids,token_type_ids=Plus().encode(text=batch['text'],tokenizer=tokenizer,max_length=64)

                # labels_ids,token_labels_ids=Plus().encode(text=str(batch['labels']),tokenizer=tokenizer,max_length=3)      
                # labels=[batch['labels']]
                # print(labels)  
                labels_ids = torch.tensor(int(batch['labels'])) # Batch size 1  # Batch size 1
                # if torch.cuda.is_available():
                # input_ids=input_ids.to()
                inputs = {'input_ids': input_ids,
                'attention_mask': input_ids,
                # 'labels': batch['labels'],
                'labels': labels_ids,
                "token_type_ids":token_type_ids}
                # print('inputs',inputs)
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                # print(loss)

                # if args.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args['gradient_accumulation_steps'] > 1:
                    loss = loss / args['gradient_accumulation_steps']

                if args['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % args['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            # Save model checkpoint 保存模型
            # output_dir=
            # output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
            output_dir = os.path.join(args['output_dir'], 'checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(vocab_path=output_dir)
            pbar(step, {'loss': loss.item()})
            print(" ")
            # if 'cuda' in str(args.device):
            if args['device']=='cuda':
                torch.cuda.empty_cache()
        return global_step, tr_loss / global_step