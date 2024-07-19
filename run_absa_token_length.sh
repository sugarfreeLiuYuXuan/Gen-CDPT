time=$(date "+%m_%d_%H:%M")
# NUMS=6
# token_length=(40 50 60 70 80 90 100 110 120 130 140 150)
# echo "token length长度 【rest16】"
# for TL in "${token_length[@]}"; do
#     for MYCAT in $(seq 0 $((NUMS-1))); do
#          CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type "$MYCAT" --num_clusters "$NUMS" --token_length $TL --n_gpu 0 --do_train --max_seq_length 256 --E_I ALL --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16  --learning_rate 1e-4 --num_train_epochs 20
#     wait 
#     echo "完成 rest16 token length长度: $TL  的 cluster id: $MYCAT"
#     done
#     wait
#     CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --dynamic --num_clusters "$NUMS" --token_length "$TL" --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 16 --gradient_accumulation_steps 1 --eval_batch_size 16  --learning_rate 1e-4 --num_train_epochs 20
#     echo "***************************************"
#     echo "训练完成 token length长度: $TL  in rest16"
#     echo "***************************************"
# done

# echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"

# wait
token_length=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250)
# NUMS=8
# echo "token length长度 【laptop14】"
# for TL in "${token_length[@]}"; do
#     for MYCAT in $(seq 0 $((NUMS-1))); do
#          CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --cate_type "$MYCAT" --num_clusters "$NUMS" --token_length $TL --n_gpu 0 --do_train --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
#     wait 
#     echo "完成 laptop14 token length长度: $TL  的 cluster id: $MYCAT"
#     done
#     wait
#     CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --dynamic --num_clusters "$NUMS" --token_length "$TL" --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
#     echo "***************************************"
#     echo "训练完成 token length长度: $TL  in laptop14"
#     echo "***************************************"
# done

# 看看最佳token_length


token_length=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250)
echo "token length长度 【laptop14】"
for TL in "${token_length[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning  --token_length "$TL" --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
    wait 
    echo "训练完成 token length长度: $TL  in laptop14"
done

echo "token length长度 【rest16】"
for TL in "${token_length[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning  --token_length "$TL" --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20
    wait 
    echo "训练完成 token length长度: $TL  in rest16"
done


# CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --dynamic --num_clusters 6 --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# wait
# echo "done rest16 dynamic 32 batch_size"

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --use_augmentation --prefix_tuning --dynamic --num_clusters 8 --token_length 64 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# wait
# echo "done laptop14 dynamic 32 batch_size"

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset rest16 --use_sent_flag --use_prompt_flag --prefix_tuning --dynamic --num_clusters 6 --token_length 128 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# echo "done no LLM rest16"
# wait

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset laptop14 --use_sent_flag --use_prompt_flag --prefix_tuning --dynamic --num_clusters 8 --token_length 64 --n_gpu 0 --do_train --do_direct_eval --max_seq_length 256 --E_I ALL --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 32  --learning_rate 1e-4 --num_train_epochs 20

# echo "done no LLM laptop14"
# wait





