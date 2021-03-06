from albert_pytorch import *

def t_classifier():
    model_name_or_path="prev_trained_model/albert_tiny"
    P=Plus()
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    P.args['class_name']="AlbertForSequenceClassification"
    model,tokenizer,config_class=P.load_model()

    train_dataloader=[{"text":"柯基犬是个狗子","labels":1},{"text":"柯基犬喜欢打架","labels":1},{"text":"哈士奇是个狗子","labels":0}]
    P.args['num_train_epochs']=20
    P.train(train_dataloader=train_dataloader, model=model, tokenizer=tokenizer)

def t_mlm():

    P=Plus()
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    P.args['class_name']="AlbertForMaskedLM"
    model,tokenizer,config_class=P.load_model()

    train_dataloader=[{"text":"柯基犬是个狗子","labels":1},{"text":"柯基犬喜欢打架","labels":1},{"text":"哈士奇是个狗子","labels":0}]
    # P.args['num_train_epochs']=20
    # P.train(train_dataloader=train_dataloader, model=model, tokenizer=tokenizer)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    for batch in train_dataloader:
        
        input_ids = torch.tensor(tokenizer.encode(batch['text'])).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
        # print(loss, prediction_scores)
        # p=torch.argmax(prediction_scores).item()
        # masked_index=1

        for masked_index in range(len(batch['text'])):
            # predicted_index = torch.argmax(prediction_scores[0, i]).item()
            # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
            # print(predicted_token)        
            k = 10
            probs, indices = torch.topk(torch.softmax(prediction_scores[0, masked_index], -1), k)
            predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
            print(predicted_tokens)


def t_mlm_pre(text):
    """
    预测mask数据
    """

    P=Plus()
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    P.args['class_name']="AlbertForMaskedLM"
    model,tokenizer,config_class=P.load_model()

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]
    # print(loss, prediction_scores)
    # p=torch.argmax(prediction_scores).item()
    # masked_index=1

    # for masked_index in range(len(text)):        
    #     k = 10
    #     probs, indices = torch.topk(torch.softmax(prediction_scores[0, masked_index], -1), k)
    #     predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    #     print(predicted_tokens)
    masked_index =10
    k = 10
    probs, indices = torch.topk(torch.softmax(prediction_scores[0, masked_index], -1), k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    print(predicted_tokens)
# t_mlm()
text="[CLS] 今天将第一个维度消除，也就是将两个[3*4]矩阵只保留一个， [SEP] 因此要在两组中作比较，即将上下两个[3*4]的 [MASK]   [SEP]"
t_mlm_pre(text)


def t_wire(text):
    """
    自动写作
    """
    model_name_or_path="prev_trained_model/albert_tiny"
    P=Plus()
    P.args['class_name']="AlbertForMaskedLM"
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    model,tokenizer,config_class=P.load_model()
      
    input_ids = torch.tensor(tokenizer.encode(text+" [MASK]  ")).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]

    masked_index=len(text)
    k = 10
    probs, indices = torch.topk(torch.softmax(prediction_scores[0, masked_index], -1), k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    print(predicted_tokens)
    print('-' * 50)
    # text.split("")
    for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
        # text[masked_index] = t
        print(t)
        # print(tokenized_text[masked_index],"===>>",t)
        print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), text[:100]), '...')
    return predicted_tokens
# text="今天天气还不错"
# for i in range(10):
#     w= t_wire(text)
#     text=text+w[0]
# print(text)



def t_ner(text):
    """
    标记
    """
    model_name_or_path="prev_trained_model/albert_tiny"
    P=Plus()
    P.args['class_name']="AlbertForTokenClassification"
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    model,tokenizer,config_class=P.load_model()
      
    # input_ids = torch.tensor(tokenizer.encode(text+" [MASK]  ")).unsqueeze(0)  # Batch size 1
    input_ids=P.encode_one(text,tokenizer,max_length=512)

    print("input_ids",input_ids)
    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
    print("labels",labels)
    outputs = model(input_ids, labels=labels)
    loss, scores = outputs[:2]
    # print(scores[0])
    # print(loss)
    # print(seq_relationship_scores)
    # print( torch.argmax(seq_relationship_scores).item())

    # for i in range(len(t))
    # np.argmax(predictions[i], axis=1).flatten()
    print(torch.argmax(scores[0],axis=1).numpy().tolist())
    return torch.argmax(scores[0],axis=1).numpy().tolist()



def get_special_tokens_mask(text):
    """
    获取标记的信息
    """
    model_name_or_path="prev_trained_model/albert_tiny"
    P=Plus()
    P.args['class_name']="AlbertForTokenClassification"
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    model,tokenizer,config_class=P.load_model()
      
    # input_ids = torch.tensor(tokenizer.encode(text+" [MASK]  ")).unsqueeze(0)  # Batch size 1
    # input_ids=P.encode_one(text,tokenizer,max_length=512)
    # input_ids=P.mask(text,tokenizer)

    # print("input_ids",input_ids)
    # labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids, labels=labels)
    # loss, scores = outputs[:2]
    # # print(scores[0])
    # # print(loss)
    # # print(seq_relationship_scores)
    # # print( torch.argmax(seq_relationship_scores).item())

    # # for i in range(len(t))
    # # np.argmax(predictions[i], axis=1).flatten()
    # print(torch.argmax(scores[0],axis=1).numpy().tolist())
    # return torch.argmax(scores[0],axis=1).numpy().tolist()



def t_mlm_train():
    print("t_mlm_train 训练mlm模型")

    P=Plus()
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    P.args['class_name']="AlbertForMaskedLM"
    model,tokenizer,config_class=P.load_model()

    train_dataloader=[{"text":"柯基犬是个狗子","labels":1},{"text":"柯基犬喜欢打架","labels":1},{"text":"哈士奇是个狗子","labels":0}]
    data=P.load_and_cache_examples('terry',tokenizer)
    P.train(train_dataset=data,model=model, tokenizer=tokenizer)

# t_mlm_train()



# def load_data(task_name='terry'):
#     P=Plus()
#     P.args['class_name']="AlbertForMaskedLM"
#     P.args['model_name_or_path']="prev_trained_model/albert_tiny"
#     model,tokenizer,config_class=P.load_model()
#     P.args['max_seq_length']=50
#     # for all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels in   P.load_data('terry',tokenizer):
#     #     # print(all_input_ids.numpy().tolist())
#     #     predicted_tokens = tokenizer.convert_ids_to_tokens(all_input_ids.numpy().tolist())
#     #     print(predicted_tokens)
#         # predicted_tokens = tokenizer.convert_ids_to_tokens(all_token_type_ids.numpy().tolist())
#         # print(predicted_tokens)
#     data=P.load_data(task_name,tokenizer)
#     P.train(data, model, tokenizer)
# text="[CLS] 今天将第一个维度消除，也就是将两个[3*4]矩阵只保留一个， [SEP] 因此要在两组中作比较，即将上下两个[3*4]的 [MASK]   [SEP]"
# t_ner(text)


def load_data(task_name='terry'):
    P=Plus()
    P.args['class_name']="AlbertForTokenClassification"
    P.args['model_name_or_path']="prev_trained_model/albert_tiny"
    P.args['data_dir']="dataset/terryner"
    model,tokenizer,config_class=P.load_model()
    P.args['max_seq_length']=50
    P.args['num_train_epochs']=10
    P.args['train_batch_size']=30
    P.args['num_labels']=10
    # for all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels in   P.load_data(task_name,tokenizer):
    #     print("len1",len(all_input_ids))
    #     print(all_input_ids)
    #     print("len2",len(all_labels))
    #     print(all_labels)
    #     # print(all_input_ids.numpy().tolist())
    #     predicted_tokens = tokenizer.convert_ids_to_tokens(all_input_ids.numpy().tolist())
    #     print(predicted_tokens)
    #     # predicted_tokens = tokenizer.convert_ids_to_tokens(all_token_type_ids.numpy().tolist())
    #     # print(predicted_tokens)
    data=P.load_data(task_name,tokenizer)
    P.train(train_dataset=data, model=model, tokenizer=tokenizer)


# get_special_tokens_mask(text)
# load_data("terryner")
 
# print(tokenizer.encode(text+" [MASK]  "))
# P=Plus()
# P.mask(text)