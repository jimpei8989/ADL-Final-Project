# run prediction
python3.8 src/predict_dst.py --model_name_or_path models/bert-dg-special-tokens --pretrained_dir ckpt/DST/bert-dg-special-tokens_expand/checkpoint-6420 --train_args_path ckpt/DST/bert-dg-special-tokens_expand/arguments.json --gpu_id 0 --test_data_dir "${1}"  --prediction_csv "${2}" --max_span_length 16
