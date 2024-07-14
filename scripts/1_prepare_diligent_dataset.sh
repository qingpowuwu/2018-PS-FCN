#!/bin/bash

# 设置工作目录
cd /home/qingpowuwu/Project_15_illumination/4_PS-FCN-master-2018

# 创建必要的目录结构
mkdir -p data/datasets
cd data/datasets

# 设置数据集名称
name="DiLiGenT"

# 创建一个符号链接，将实际的数据集目录链接到当前目录下的 "DiLiGenT" 文件夹。
ln -s /home/qingpowuwu/Project_15_illumination/0_Dataset_Original/DiLiGenT ${name}

# 进入数据集目录
cd ${name}/pmsData/

# 生成 objects.txt 文件（排除 objects.txt 本身）
# 列出当前目录下的所有文件和文件夹，排除 objects.txt 本身，然后将结果保存到 objects.txt 文件中。
ls | sed '/objects.txt/d' > objects.txt

# 复制 filenames.txt
# 将 ballPNG 目录下的 filenames.txt 文件复制到当前目录。
cp ballPNG/filenames.txt .

# 返回到根目录
cd ../../../../

echo "操作完成。符号链接已创建，objects.txt 已更新，filenames.txt 已复制。"