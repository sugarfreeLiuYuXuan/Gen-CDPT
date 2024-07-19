current_dataset=rest16
cluster=30
for MYCAT in $(seq 0 $((MYNUM-1))); do
        CUDA_VISIBLE_DEVICES=0 python main.py   --dataset $current_dataset \
        									    --use_sent_flag \
        									    --use_prompt_flag \
        									    --use_augmentation \
            									--prefix_tuning \
            									--cate_type "$MYCAT" \
            									--num_clusters $cluster \
            									--token_length 80 \
            									--n_gpu 0 \
            									--do_train \
            									--max_seq_length 256 \
            									--E_I ALL \
            									--train_batch_size 32 \
            									--gradient_accumulation_steps 1 \
            									--eval_batch_size 32  \
            									--learning_rate 1e-4 \
            									--num_train_epochs 20
done