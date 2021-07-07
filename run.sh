SCHEMA_JSON=$1
TEST_DATA_DIR=$2
PREDICTION_CSV=$3

# run prediction
python3.8 src/predict_dst.py \
    --model_name_or_path models/bert-dg-special-tokens \
    --pretrained_dir ckpt/DST/bert-dg-special-tokens_expand/checkpoint-6420 \
    --train_args_path ckpt/DST/bert-dg-special-tokens_expand/arguments.json \
    --gpu_id 0 --max_span_length 16 \
    --schema_json ${SCHEMA_JSON} --test_data_dir ${TEST_DATA_DIR}  --prediction_csv ${PREDICTION_CSV} 

