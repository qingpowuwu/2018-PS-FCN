# Abstract
总共有4个contributions:
1. 之前的工作不能


## Model 的 架构分析

```
=> using pre-trained model data/models/PS-FCN_B_S_32.pth.tar
PS_FCN(
  (extractor): FeatExtractor(
    (conv1): Sequential(
      (0): Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv3): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv4): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv5): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv6): Sequential(
      (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (conv7): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
  )
  (regressor): Regressor(
    (deconv1): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (deconv2): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (deconv3): Sequential(
      (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (est_normal): Sequential(
      (0): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
  )
)
=> Model Parameters: 2210048
```

这段输出展示了一个名为PS_FCN的神经网络模型的结构。这是一个用于光度立体视觉（Photometric Stereo）的全卷积网络（Fully Convolutional Network）：

1. 整体结构：
   PS_FCN 模型主要由两个部分组成：特征提取器（extractor）和回归器（regressor）。

2. 特征提取器 (FeatExtractor):
   - 包含7个卷积层（conv1到conv7）
   - 输入通道数为6（可能是3通道图像加3通道光照信息）
   - 使用了卷积（Conv2d）、转置卷积（ConvTranspose2d）和LeakyReLU激活函数
   - 网络结构呈现先降采样后升采样的形式，有助于捕获多尺度特征

3. 回归器 (Regressor):
   - 包含4个卷积层（deconv1到deconv3，以及est_normal）
   - 主要用于将提取的特征映射到最终的输出（可能是表面法线）
   - 同样使用了卷积、转置卷积和LeakyReLU

4. 具体层结构：
   - 大多数卷积层后面跟着LeakyReLU激活函数（负斜率为0.1）
   - 使用了填充（padding）来保持特征图大小
   - conv2和conv4使用了步长为2的卷积，用于降采样
   - conv6和deconv3使用了转置卷积进行上采样

5. 最后一层（est_normal）：
   - 输出3个通道，可能对应于表面法线的x、y、z分量

6. 模型参数：
   - 总参数数量为2,210,048

这个模型设计用于从多张具有不同光照条件的图像中估计物体表面的法线。特征提取器负责从输入图像中提取有用的特征，而回归器则将这些特征转换为表面法线估计。

模型使用了全卷积的设计，这意味着它可以处理任意大小的输入图像，并产生相应大小的输出。LeakyReLU的使用有助于解决传统ReLU可能面临的"死亡ReLU"问题，使得模型训练更加稳定。

