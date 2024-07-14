# Abstract

总共有4个contributions:
1. 之前的工作不能泛化到 real-world objects (DiLiGenT, Gourd&Apple, Light Stage Data Gallery)
2. 我们提出了 PS-FCN: 输入：同1个物体，在不同光照下的拍出来的 2d_images, 输出normal map：

   <img width="626" alt="Screenshot 2024-07-14 at 11 10 10 PM" src="https://github.com/user-attachments/assets/96a13798-6228-44ef-9921-7d9de2d948c4">

4. 我们的输入不需要是 pre-defined set of light directions => 所以是 order-agnostic 的
5. 尽管我们训练的时候用的是 synthetic 数据 (blobby & sculpture), 但是 测试的时候 除了使用了 synthetic 数据 (Sphere & Bunny)。我们还在 real 数据 (DiLiGenT, Gourd&Apple, Light Stage Data Gallery) 上做了测试：这里 blobby & sculpture 数据集 的 normal map 是我们 在之前工作提出的 shape 自己计算出来的

# Dataset

## Train Dataset (Blobby: 25920个 2d images & Sculpture: 592920个 2d images)

Train Dataset 用的是 Blobby & Sculpture 数据集

<img width="702" alt="Screenshot 2024-07-14 at 11 12 30 PM" src="https://github.com/user-attachments/assets/eb7c978b-f1f8-4de3-8aa7-393174a0eca7">

* 我们采用了 [10], [11] 的 3d 模型 & [38] 的 raytracer 方法来 => 生成 => 对应的 normal map

### (1) Blobby Dataset

我们 follow [8] 来渲染 blobby 数据集中的 10个 shapes，每个 shape 都有 1296 个 2d images

* => 样本量：25920 个 2d images

### (2) Sculpture Dataset

因为 blobby 数据集 的表面太 smooth 了 & 缺少 details，所以我们 还加入了 sculpture 数据集 来训练我们的模型

* => 样本量：59292 个 2d images

## Test Dataset (Blobby: 25920个 2d images & Sculpture: 592920个 2d images)

Test Dataset 用的是 1个 synthetic dataset: Sphere & Bunny 数据集 和 3个 real dataset: DiLiGentT、 Gourd&Apple、Light Stage Data Gallery 数据集

### Synthetic Dataset (200 个 2d images)

Synthetic Dataset 包括 Sphere & Bunny 2种 shape: 每个 shape 都用 100个 2d-images

### Real Dataset （2026 个 2d images）

Real Dataset 包括 3个 non-Lambertian 的 real 数据集:
1. DiLiGentT: 有10个 shape，每个 shape 都有 96个 2d-images => 样本量：196 个 2d images
2. Gourd&Apple: 有3个 shape，有 102+98+112个 = 312 个 2d-images => 样本量：312 个 2d images
3. Light Stage Data Gallery: 有 6个 shape，每个shape有 253 个 2d-images => 样本量: 1518 个 2d images

## Model 的 架构分析
<img width="830" alt="Screenshot 2024-07-14 at 11 11 22 PM" src="https://github.com/user-attachments/assets/3b92c9ca-584e-47b0-bedf-16def8ac391a">

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

# Experiments
## Metrics: 采用了 Mean angular error (MAE)
## Results
### Results on Sphere 数据集
我们对比了PS-FCN 和 baseline[1] 在 Sphere 数据集上 MAE 的表现：
<img width="1158" alt="Screenshot 2024-07-14 at 11 24 50 PM" src="https://github.com/user-attachments/assets/d769564d-b459-4605-962c-fcb95a1e9d1f">
这里 蓝色是 baseline[1] 的表现，黑色是我们 PS-FCN 的表现

### Results on DiLiGenT 数据集
我们对比了PS-FCN 和 baseline[1] 和其他工作在 DiLiGenT 数据集上 MAE 的表现：
<img width="816" alt="Screenshot 2024-07-14 at 11 26 29 PM" src="https://github.com/user-attachments/assets/5d089c90-78ae-4e7d-a330-b2aa3ddf83b7">
我们发现，我们的 PS-FCN 的 MAE 能够降低到 8.39

### Visualization on DiLiGenT 数据集

<img width="629" alt="Screenshot 2024-07-14 at 11 28 44 PM" src="https://github.com/user-attachments/assets/04ed9754-ef5b-4e56-9016-5124b09114ef">

这里可视化了我们的 PS-FCN 在 DiLiGenT 数据集 上预测出来的 normal maps
