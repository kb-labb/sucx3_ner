#! /usr/bin/bash

# Data directory
data="./data"
mkdir  $data
mkdir $data/raw

# Download SUC 3.0
wget http://spraakbanken.gu.se/lb/resurser/meningsmangder/suc3.xml.bz2 -P $data/raw
cd $data/raw
bunzip2 suc3.xml.bz2
cd -

# convert xml to conll format
mkdir $data/base
OT="original_tags"
ST="simple_tags"
STNMT="simple_tags.no_MSR_TME"
python suc_handler.py --infile $data/raw/suc3.xml --outfile $data/base/suc3.$OT.conll --original_tags
python suc_handler.py --infile $data/base/suc3.$OT.conll --outfile $data/base/suc3.$OT.jsonl --conllu2json

python suc_handler.py --infile $data/raw/suc3.xml --outfile $data/base/suc3.$ST.conll
python suc_handler.py --infile $data/base/suc3.$ST.conll --outfile $data/base/suc3.$ST.jsonl --conllu2json

python suc_handler.py --infile $data/raw/suc3.xml --outfile $data/base/suc3.$STNMT.conll --no_MSR_TME
python suc_handler.py --infile $data/base/suc3.$STNMT.conll --outfile $data/base/suc3.$STNMT.jsonl --conllu2json

# create canonical split based on original tag distribution
mkdir $data/$OT
python split_and_filter.py --infile $data/base/suc3.$OT.jsonl  --outfolder $data/$OT --seed 12345


# recreate canonical split based on original tag distribution's split_ids
mkdir $data/$ST
python split_and_filter.py --infile $data/base/suc3.$ST.jsonl  --outfolder $data/$ST --split_ids $data/$OT/split_ids.json

mkdir $data/$STNMT
python split_and_filter.py --infile $data/base/suc3.$STNMT.jsonl  --outfolder $data/$STNMT --split_ids $data/$OT/split_ids.json

# lowercase all or only NE
for i in $OT $ST $STNMT;
do
    F=$data/$i;

    x="lower"
    mkdir $F/$x
    T=$F/$x;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl;

    mkdir $F/${x}_mix
    T=$F/${x}_mix;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl --mix --seed 12345;

    mkdir $F/${x}_both
    T=$F/${x}_both;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl --both;

    x="ne_lower"
    mkdir $F/$x
    T=$F/$x;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl --ne_only;

    mkdir $F/${x}_both
    T=$F/ne_lower_both;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl --ne_only --both;

    mkdir $F/${x}_mix
    T=$F/${x}_mix;
    python lowercase_ner_data.py --infiles $F/train.jsonl $F/dev.jsonl $F/test.jsonl --outfiles $T/train.jsonl $T/dev.jsonl $T/test.jsonl --ne_only --mix --seed 12345;
done

