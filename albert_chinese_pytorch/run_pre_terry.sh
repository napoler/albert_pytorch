CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_tiny
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="terry"

python run_pretraining.py \
    --data_dir=dataset/ \
    -- model_path=prev_trained_model/albert_tiny/ \
    --vocab_path=prev_trained_model/albert_tiny/vocab.txt \
    --data_name=albert \
    --config_path=prev_trained_model/albert_tiny/config.json \
    --output_dir=outputs/t \
    --data_name=albert \
    --share_type=all



# python run_classifier.py \
#   --model_type=albert \
#   --model_name_or_path=$BERT_BASE_DIR \
#   --task_name=$TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir=$DATA_DIR/${TASK_NAME}/ \
#   --max_seq_length=128 \
#   --per_gpu_train_batch_size=64 \
#   --per_gpu_eval_batch_size=64 \
#   --learning_rate=1e-4 \
#   --num_train_epochs=5.0 \
#   --logging_steps=3731 \
#   --save_steps=3731 \
#   --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#   --overwrite_output_dir
