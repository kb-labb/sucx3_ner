#!/usr/bin/bash

which python
/bin/hostname -s

DATA_VAR="original_tags"
TUNE_ALG="PBT"  # BOHB
dv=$(echo ${DATA_VAR} | sed 's/\//_/g')
TUNE_NAME="${dv}_${TUNE_ALG}"

# DATA_SUFFIX=""
# --train_file ./data/martin_data/train.lower.only.jsonl \
  #                  --validation_file ./data/martin_data/dev.lower.only.jsonl \
  #                  --test_file ./data/martin_data/test.lower.only.jsonl \

run_cmd="python run_ner.py --model_name_or_path KB/bert-base-swedish-cased \
                  --train_file ./data/${DATA_VAR}/train.jsonl \
                  --validation_file ./data/${DATA_VAR}/dev.jsonl \
                  --test_file ./data/${DATA_VAR}/test.jsonl \
                  --output_dir KB-BERT-ner-regular-tune \
                  --do_train \
                  --do_eval \
                  --do_predict \
                  --task_name ner \
                  --cache_dir models \
                  --return_entity_level_metrics 0 \
                  --per_device_train_batch_size 60 \
                  --per_device_eval_batch_size 60 \
                  --overwrite_output_dir \
                  --gradient_accumulation_steps 1
                  --num_train_epochs 5 \
                  --evaluation_strategy steps \
                  --save_strategy steps \
                  --skip_memory_metrics \
                  --eval_steps 200 \
                  --fp16 \
                  --disable_tqdm 1 \
                  --tune ${TUNE_NAME} \
                  --tune_alg ${TUNE_ALG} \
                  --tune_trials 100 \
                  --tune_local_dir ./ray_results/
                  "
                  # --eval_steps 10000 \
                  # --save_steps 10000 \
                  # --max_train_samples 3000
                  # --max_val_samples 100
                  # --max_test_samples 100
                  # --do_eval \

                  #
                  # --tune regular_lower_only \
                  #--train_file ./data/suc_train.both.jsonl \
                  #--validation_file ./data/suc_dev.both.jsonl \
                  #--test_file ./data/suc_test.both.jsonl \
                  #--learning_rate 0.000836949 \
                  #--weight_decay 0.0727959 \

                  # --learning_rate 2.2199e-05
                  # --weight_decay 0.015191

                  # --learning_rate 9.808539615186473e-06
                  # --weight_decay 0.06887642560886498

echo $run_cmd
$run_cmd