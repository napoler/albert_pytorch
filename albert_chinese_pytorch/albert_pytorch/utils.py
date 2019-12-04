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

