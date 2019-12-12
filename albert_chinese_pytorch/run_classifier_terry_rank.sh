CURRENT_DIR=`pwd`
# export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_tiny
#预训练模型位置
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/terry_rank_output
#数据位置
export DATA_DIR=$CURRENT_DIR/dataset
#输出位置
export OUTPUR_DIR=$CURRENT_DIR/outputs
# 任务名称
TASK_NAME="terry_rank"

#数据类型
TASK_NAME_DF="terry"
#   python run_classifier.py \
#     --model_type=albert \
#     --model_name_or_path=$BERT_BASE_DIR \
#     --task_name=$TASK_NAME_DF \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --data_dir=$DATA_DIR/${TASK_NAME}/ \
#     --max_seq_length=32 \
#     --per_gpu_train_batch_size=2048 \
#     --per_gpu_eval_batch_size=1512 \
#     --learning_rate=1e-8 \
#     --num_train_epochs=20.0 \
#     --logging_steps=3731 \
#     --save_steps=3731 \
#     --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#     --overwrite_output_dir

#进行无限次数循环 直到选择结束为止
while :
do
    python run_classifier.py \
      --model_type=albert \
      --model_name_or_path=$BERT_BASE_DIR \
      --task_name=$TASK_NAME_DF \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=32 \
      --per_gpu_train_batch_size=1512 \
      --per_gpu_eval_batch_size=1512 \
      --learning_rate=1e-8 \
      --num_train_epochs=20.0 \
      --logging_steps=3731 \
      --save_steps=3731 \
      --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir

    #自动复制备份
    
    # # mkdir $CURRENT_DIR/prev_trained_model/terry_rank_output.bak
    # mv  $BERT_BASE_DIR  $BERT_BASE_DIR.bak

    export TIME=`date +%Y%m%d%H%M%S`
    # # export BACKUP= $CURRENT_DIR/prev_trained_model/backup/${TASK_NAME}/${TIME}
    # # mkdir $CURRENT_DIR/prev_trained_model/backup/

    
      if [ ! -x "$CURRENT_DIR/prev_trained_model/backup/" ];then 
      mkdir $CURRENT_DIR/prev_trained_model/backup/
      fi 

    mkdir $CURRENT_DIR/prev_trained_model/backup/${TASK_NAME}_${TIME}
    cp -r $BERT_BASE_DIR $CURRENT_DIR/prev_trained_model/backup/${TASK_NAME}_${TIME}

    if [ ! -x "$BERT_BASE_DIR" ];then 
    mkdir $BERT_BASE_DIR
    fi 
    
    cp outputs/${TASK_NAME}_output/* $BERT_BASE_DIR
    echo "休息10s"
    sleep 10

done



# # python run_classifier.py \
# #   --model_type=albert \
# #   --model_name_or_path=$BERT_BASE_DIR \
# #   --task_name=$TASK_NAME \
# #   --do_train \
# #   --do_eval \
# #   --do_lower_case \
# #   --data_dir=$DATA_DIR/${TASK_NAME}/ \
# #   --max_seq_length=128 \
# #   --per_gpu_train_batch_size=64 \
# #   --per_gpu_eval_batch_size=64 \
# #   --learning_rate=1e-4 \
# #   --num_train_epochs=5.0 \
# #   --logging_steps=3731 \
# #   --save_steps=3731 \
# #   --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
# #   --overwrite_output_dir
