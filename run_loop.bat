@echo on
@echo off

set cate_types=1 2 3 4 5

for %%A in (%cate_types%) do (
    echo Running with cate_types %%A
    D:\Anconda\envs\absa\python.exe main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type %%A --num_clusters 6 --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 8 --gradient_accumulation_steps 1 --eval_batch_size 8  --learning_rate 1e-4 --num_train_epochs 20
    echo Done with cate_types %%A
)