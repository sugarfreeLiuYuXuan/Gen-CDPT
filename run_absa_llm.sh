
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag  --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# wait
# echo "完成base rest16"

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning  --num_clusters 6 --dynamic --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# # echo "完成 no LLM rest16"
# # wait

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# wait
# echo "完成base laptop14"

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --prefix_tuning --token_length 64 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# echo "done no LLM laptop14"


# cluster=(12)

# wait
# echo "begin to train cluster 【laptop14】"
# for MYNUM in "${cluster[@]}"; do
#     for MYCAT in $(seq 0 $((MYNUM-1))); do
#         CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type "$MYCAT" --num_clusters "$MYNUM" --token_length 80 --n_gpu 0 --do_train --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
#     wait
#     echo "----------------------------------------"
#     echo "num_clusters: $MYNUM  cate_type: $MYCAT"
#     echo "----------------------------------------"
#     done
#     CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag  --use_augmentation --prefix_tuning --dynamic --num_clusters "$MYNUM" --token_length 80 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
#     wait
#     echo "***************************************"
#     echo "ok dynamic"
#     echo "***************************************"
# done
# wait


CUDA_VISIBLE_DEVICES=0 python main.py \
                    --dataset rest16 \ 
                    --use_sent_flag \
                    --use_prompt_flag \
                    --use_augmentation \
                    --prefix_tuning  \
                    --token_length 64 \
                    --n_gpu 0 \
                    --do_direct_eval \
                    --max_seq_length 256 \
                    --E_I ALL \
                    --train_batch_size 32 \
                    --gradient_accumulation_steps 1 \
                    --eval_batch_size 32  \
                    --learning_rate 1e-4 \
                    --num_train_epochs 20