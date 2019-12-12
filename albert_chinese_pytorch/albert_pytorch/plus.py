from .model.modeling_albert import *
from .model.tokenization_bert import BertTokenizer


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
        pass
    def load_model(self,class_name,model_path):
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
        model_name_or_path= model_path
        config_class, model_class, tokenizer_class = MODEL_CLASSES[class_name]
        tokenizer = tokenizer_class.from_pretrained(model_path,
                                                    do_lower_case=False)
        model = model_class.from_pretrained(model_path)
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
