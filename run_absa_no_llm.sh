
CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

echo "done no LLM rest16"
wait

CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --prefix_tuning --token_length 64 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

echo "done no LLM laptop14"