# remove duplicates
cmd="python compare_data.py \
            --remove_doubles \
            --files_a data/martin_data/train.jsonl data/martin_data/dev.jsonl data/martin_data/test.jsonl \
            --files_b new_data/martin_data/train.jsonl new_data/martin_data/dev.jsonl new_data/martin_data/test.jsonl"
cmd="python compare_data.py \
            --remove_doubles \
            --files_a data/suc_train.jsonl data/suc_dev.jsonl data/suc_test.jsonl \
            --files_b new_data/train.jsonl new_data/dev.jsonl new_data/test.jsonl"
#$cmd

data="./new_data"
for data in "./data" "./data/martin_data";
do
    suffix="lower.both"
    cmd="python lowercase_ner_data.py \
                --infiles  ${data}/train.jsonl ${data}/dev.jsonl ${data}/test.jsonl \
                --outfiles ${data}/train.${suffix}.jsonl ${data}/dev.${suffix}.jsonl ${data}/test.${suffix}.jsonl \
                --both"
    $cmd

    suffix="lower.only"
    cmd="python lowercase_ner_data.py \
                --infiles  ${data}/train.jsonl ${data}/dev.jsonl ${data}/test.jsonl \
                --outfiles ${data}/train.${suffix}.jsonl ${data}/dev.${suffix}.jsonl ${data}/test.${suffix}.jsonl"
    $cmd

    suffix="ne_lower.both"
    cmd="python lowercase_ner_data.py \
                --infiles  ${data}/train.jsonl ${data}/dev.jsonl ${data}/test.jsonl \
                --outfiles ${data}/train.${suffix}.jsonl ${data}/dev.${suffix}.jsonl ${data}/test.${suffix}.jsonl \
                --both \
                --ne_only"
    $cmd

    suffix="ne_lower.only"
    cmd="python lowercase_ner_data.py \
                --infiles  ${data}/train.jsonl ${data}/dev.jsonl ${data}/test.jsonl \
                --outfiles ${data}/train.${suffix}.jsonl ${data}/dev.${suffix}.jsonl ${data}/test.${suffix}.jsonl \
                --ne_only"
    $cmd
done