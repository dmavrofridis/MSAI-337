#!/bin/bash
# Get the variables required from the yaml file
export TRAIN_FILE=./files/data/output/woz.train_b.txt
export TEST_FILE=./files/data/output/woz.test_b.txt
python ./run_clm.py --output_dir=./files/data/output/b --model_type=gpt2 --model_name_or_path=gpt2 --train_file=$TRAIN_FILE --do_train --validation_file=$TEST_FILE --do_eval --save_total_limit 10 --num_train_epochs 10
