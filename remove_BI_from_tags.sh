#! /usr/bin/bash

# first argument to script is the data folder
DATA=$1
for D in simple_tags simple_tags.no_MSR_TME;
do
    OLD_DIR="${DATA}/${D}"
    NEW_DIR="${DATA}/really_${D}"

    if [[ -d "$NEW_DIR" ]]
    then
        echo "$NEW_DIR exists on your filesystem."
    else
        echo "$NEW_DIR does not exist"
        mkdir $NEW_DIR;
    fi
    echo "base"
    for x in train dev test;
    do
        python simplify_tags.py --infile $OLD_DIR/$x.jsonl --outfile $NEW_DIR/$x.jsonl
    done

    for V in lower lower_mix lower_both ne_lower ne_lower_both ne_lower_mix;
    do
        OLD_DIR_V="${OLD_DIR}/${V}"
        NEW_DIR_V="${NEW_DIR}/${V}"

        echo $V

        if [[ -d "$NEW_DIR_V" ]]
        then
            echo "$NEW_DIR_V exists on your filesystem."
        else
            echo "$NEW_DIR_V does not exist"
            mkdir $NEW_DIR_V;
        fi

        for x in train dev test;
        do
            python simplify_tags.py --infile $OLD_DIR_V/$x.jsonl --outfile $NEW_DIR_V/$x.jsonl
        done
    done
 done   