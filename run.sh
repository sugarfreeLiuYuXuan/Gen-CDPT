#!/usr/bin/env bash
time=$(date "+%m_%d_%H:%M")
# current_dataset=laptop14
current_dataset=rest16
CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16  --learning_rate 1e-4 --num_train_epochs 20 

# >> ./log/$current_dataset/$current_dataset_$time.log
#             --prefix_tuning \
python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20 

python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type 0 --num_clusters 6 --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20

python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --dynamic --num_clusters 6 --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20

python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type 4 --num_clusters 8 --token_length 64 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20

python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --token_length 128 --n_gpu 0 --do_inference --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20 
(5e-5 30)

python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --token_length 256 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 5e-5 --num_train_epochs 30 