#!/bin/bash

# 设置GPU索引
GPU_INDEX=0

# 循环检测条件
while true; do
    # 获取GPU信息
    GPU_INFO=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)

    # 提取Volatile GPU-Util字段的值
    GPU_UTIL=$(echo "$GPU_INFO" | awk -F',' '{print $2}' | tr -d '[:space:]')

    # 打印GPU使用情况
    echo "GPU-$GPU_INDEX Utilization: $GPU_UTIL%"

    # 判断GPU利用率是否小于10%
    if [ "$GPU_UTIL" -lt 10 ]; then
        # 等待5分钟
        sleep 300

        # 再次获取GPU信息
        GPU_INFO=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)

        # 提取新的Volatile GPU-Util字段的值
        NEW_GPU_UTIL=$(echo "$GPU_INFO" | awk -F',' '{print $2}' | tr -d '[:space:]')

        # 判断新的GPU利用率是否依然小于10%
        if [ "$NEW_GPU_UTIL" -lt 10 ]; then
            # 运行Python文件
            bash run.sh

            # 跳出循环
            break
        fi
    fi

    # 等待一段时间后重新检测
    sleep 5
done
