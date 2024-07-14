import sys, os, shutil
import torch
sys.path.append('.')

import test_utils

from datasets import custom_data_loader
from options  import run_model_opts
from models import custom_model
from utils  import logger, recorders

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args) # Namespace(benchmark='DiLiGenT_main', bm_dir='data/datasets/DiLiGenT/pmsData', cuda=True, epochs=30, fuse_type='max', in_img_num=96, in_light=True, item='calib', model='PS_FCN_run', normalize=False, resume=None, retrain='data/models/PS-FCN_B_S_32.pth.tar', run_model=True, save_root='data/Training/', seed=0, start_epoch=1, test_batch=1, test_disp=1, test_intv=1, test_save=1, time_sync=False, train_img_num=32, use_BN=False, workers=8)

def main(args):
    # 加载测试数据
    test_loader = custom_data_loader.benchmarkLoader(args)
    # 构建模型
    model = custom_model.buildModel(args)
    # 创建记录器
    recorder = recorders.Records(args.log_dir)

    # 打印 test_loader 的信息
    print('len(test_loader) = ', len(test_loader)) # 10

    # 打印 test_loader 中第一个 batch 的形状
    try:
        first_batch = next(iter(test_loader))
        if isinstance(first_batch, dict):
            for key, value in first_batch.items():
                print(f"Shape of {key} in the first batch: {value.shape}")
#                 Shape of N     in the first batch: torch.Size([1, 3, 512, 612])
                # Shape of img   in the first batch: torch.Size([1, 288, 512, 612])
                # Shape of mask  in the first batch: torch.Size([1, 1, 512, 612])
                # Shape of light in the first batch: torch.Size([1, 288, 1, 1])
        elif isinstance(first_batch, (list, tuple)):
            for i, item in enumerate(first_batch):
                print(f"Shape of item {i} in the first batch: {item.shape}")
        else:
            print(f"Shape of the first batch: {first_batch.shape}")
    except Exception as e:
        print(f"Could not retrieve first batch shape: {e}") #  'list' object has no attribute 'shape'

    # 打印 model 的信息
    print('model = ', model) # PS_FCN_run(
    # 打印 recorder 的信息
    print('recorder.log_dir = ', recorder.log_dir) # data/Training/run_model
    
    # 进行测试
    test_utils.test(args, 'test', test_loader, model, log, 1, recorder)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)



# Shape of N     in the first batch: torch.Size([1, 3, 512, 612])
# Shape of img   in the first batch: torch.Size([1, 288, 512, 612])
# Shape of mask  in the first batch: torch.Size([1, 1, 512, 612])
# Shape of light in the first batch: torch.Size([1, 288, 1, 1])

# N 张量的形状：
#     torch.Size([1, 3, 512, 612])
#     这个张量可能代表的是法线图（Normal Map）。
#     形状 [1, 3, 512, 612] 表示有 1 个样本，每个样本有 3 个通道（对应 RGB 三个通道），尺寸为 512x612 像素。

# img 张量的形状：
#     torch.Size([1, 288, 512, 612])
#     这个张量可能代表的是输入图片集合。
#     形状 [1, 288, 512, 612] 表示有 1 个样本，每个样本有 288 个通道，每个通道的尺寸为 512x612 像素。288 个通道可能表示不同光照条件下拍摄的图片。

# mask 张量的形状：
#     torch.Size([1, 1, 512, 612])
#     这个张量可能代表的是遮罩（mask），用于标记有效区域。
#     形状 [1, 1, 512, 612] 表示有 1 个样本，每个样本有 1 个通道，尺寸为 512x612 像素。遮罩通常是单通道的二进制图像。

# light 张量的形状：
#     torch.Size([1, 288, 1, 1])
#     这个张量可能代表的是光照方向信息。
#     形状 [1, 288, 1, 1] 表示有 1 个样本，每个样本有 288 个通道，每个通道代表一个光照方向。由于光照方向是一个标量（或一个小向量），所以它在空间维度上是 [1, 1]。


# 总结
# 这些张量代表了在光照方向下拍摄的图片及其相关的信息：

#     N 是法线图，描述了表面法线方向。
#     img 是在 288 个不同光照方向下拍摄的图片集合。
#     mask 是有效区域遮罩，用于标记图像中参与计算的部分。
#     light 是 288 个不同光照方向的信息。
