python run_clm.py --output_dir=files/data/output/b --model_type=gpt2 --model_name_or_path=gpt2 --train_file=files/data/output/woz.train_b.txt --do_train --validation_file=files/data/output/woz.test_b.txt --do_eval --save_total_limit 10 --num_train_epochs 10 --device=cuda --n_gpu=1