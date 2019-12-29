[**中文版说明**](./README.md) |


albert_chinese_pytorch目录下

加入flask 标记数据

宠物内容识别模型下载
https://www.kaggle.com/terrychanorg/albert-chinese-pytorch-pet
下载模型放入albert_chinese_pytorch/prev_trained_model/terry_output










转化模型
!wget https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip
python convert_albert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=prev_trained_model/albert_base_zh_tf/albert_model.ckpt --bert_config_file=../albert_config/albert_config_base.json --pytorch_dump_path=prev_trained_model/albert_base_zh/pytorch_model.bin
