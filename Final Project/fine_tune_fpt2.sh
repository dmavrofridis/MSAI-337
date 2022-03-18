# A # --output_dir=files/data/output/gpt2_untrained --model_type=gpt2 --model_name_or_path=gpt2 --validation_file=files/data/output/woz.test_a.txt --do_eval --save_total_limit 10 --per_device_train_batch_size=2 --per_device_eval_batch_size=2

# B # --output_dir=files/data/output/gpt2_trained --overwrite_output_dir --model_type=gpt2 --model_name_or_path=gpt2 --train_file=files/data/output/woz.train_b.txt --do_train --validation_file=files/data/output/woz.test_b.txt --do_eval --save_total_limit 10 --num_train_epochs 10 --per_device_train_batch_size=2 --per_device_eval_batch_size=2

# C # --output_dir=files/data/output/gpt2_fine_tuned --model_type=gpt2 --model_name_or_path=gpt2 --train_file=files/data/output/woz.train_b.txt --do_train --validation_file=files/data/output/woz.test_b.txt --do_eval --save_total_limit 10 --num_train_epochs 8 --per_device_train_batch_size=2 --per_device_eval_batch_size=2

# D # --output_dir=files/data/output/distilgpt2 --overwrite_output_dir --model_type=distilgpt2 --model_name_or_path=distilgpt2 --train_file=files/data/output/woz.train_c.txt --do_train --validation_file=files/data/output/woz.test_c.txt --do_eval --save_total_limit 10 --num_train_epochs 10 --per_device_train_batch_size=2 --per_device_eval_batch_size=2

# E # --output_dir=files/data/output/xlnet-base-cased --overwrite_output_dir --model_type=xlnet-base-cased --model_name_or_path=xlnet-base-cased --train_file=files/data/output/woz.train_c.txt --do_train --validation_file=files/data/output/woz.test_c.txt --do_eval --save_total_limit 10 --num_train_epochs 10 --per_device_train_batch_size=2 --per_device_eval_batch_size=2

# F # --output_dir=files/data/output/roberta-base --overwrite_output_dir --model_type=roberta-base --model_name_or_path=roberta-base --train_file=files/data/output/woz.train_c.txt --do_train --validation_file=files/data/output/woz.test_c.txt --do_eval --save_total_limit 10 --num_train_epochs 10 --per_device_train_batch_size=2 --per_device_eval_batch_size=2
