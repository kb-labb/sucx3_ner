cmd="python run_ner.py \
    --model_name_or_path KB/bert-base-swedish-cased-neriob \
    --train_file data/simple_tags/lower_mix/train.jsonl \
    --validation_file data/simple_tags/lower_mix/dev.jsonl \
    --test_file data/simple_tags/lower_mix/test.jsonl \
    --output_dir KB-base-swedish-cased-mega \
    --do_train \
    --do_eval \
    --do_predict \
    --cache_dir models \
    --return_entity_level_metrics 1 \
    --per_device_train_batch_size 28 \
    --per_device_eval_batch_size 28 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --skip_memory_metrics \
    --eval_steps 10000 \
    --save_total_limit 2 \
    --save_steps 10000 \
    --overwrite_output_dir \
    --fp16 
    "
$cmd