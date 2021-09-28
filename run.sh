#!/usr/bin/bash

which python
/bin/hostname -s

DATA_SUFFIX=".lower.only"
# DATA_SUFFIX=""
# --train_file ./data/martin_data/train.lower.only.jsonl \
  #                  --validation_file ./data/martin_data/dev.lower.only.jsonl \
  #                  --test_file ./data/martin_data/test.lower.only.jsonl \

run_cmd="python run_ner.py --model_name_or_path KB/bert-base-swedish-cased \
                  --train_file ./data/martin_data/train$DATA_SUFFIX.jsonl \
                  --validation_file ./data/martin_data/dev$DATA_SUFFIX.jsonl \
                  --test_file ./data/martin_data/test$DATA_SUFFIX.jsonl \
                  --output_dir KB-BERT-ner-martin-regular_lower_only \
                  --do_train \
                  --do_eval \
                  --do_predict \
                  --task_name ner \
                  --cache_dir models \
                  --return_entity_level_metrics 0 \
                  --per_device_train_batch_size 64 \
                  --per_device_eval_batch_size 64 \
                  --overwrite_output_dir \
                  --gradient_accumulation_steps 1
                  --num_train_epochs 5 \
                  --evaluation_strategy steps \
                  --eval_steps 500 \
                  --skip_memory_metrics \
                  --fp16 \
                  --disable_tqdm 1 \
                  --tune regular_lower_only \
                  "
                  #--train_file ./data/suc_train.both.jsonl \
                  #--validation_file ./data/suc_dev.both.jsonl \
                  #--test_file ./data/suc_test.both.jsonl \
                  #--learning_rate 0.000836949 \
                  #--weight_decay 0.0727959 \

echo $run_cmd
$run_cmd