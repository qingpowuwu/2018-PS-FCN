#!/bin/bash

# 设置工作目录
cd /home/qingpowuwu/Project_15_illumination/4_PS-FCN-master-2018

# 创建必要的目录结构
mkdir -p data/datasets
cd data/datasets

# 创建符号链接到已解压的数据集
for dataset in "PS_Sculpture_Dataset" "PS_Blobby_Dataset"; do
    echo "Creating symlink for $dataset"
    ln -s /home/qingpowuwu/Project_15_illumination/0_Downloaded_Original/PS-FCN/$dataset .
done

# 返回到根目录
cd ../../

echo "操作完成。符号链接已创建，指向 PS_Sculpture_Dataset 和 PS_Blobby_Dataset。"