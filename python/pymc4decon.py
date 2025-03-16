import pymc as pm
import numpy as np
import arviz as az
import aesara.tensor as at

# 假设我们有一个ground truth图像gt_image和观测到的图像obs_image
# 这里我们用随机数据来模拟
np.random.seed(42)
gt_image = np.random.rand(10, 10)  # 真实图像
psf = np.exp(-np.linspace(-1, 1, 10)**2)  # 高斯型点扩散函数
psf = psf / psf.sum()  # 归一化

# 卷积操作
from scipy.signal import convolve2d
conv_image = convolve2d(gt_image, psf[:, None], mode='same')

# 加入泊松噪声
obs_image = np.random.poisson(conv_image)

# 使用PyMC定义模型
with pm.Model() as model:
    # 定义先验分布
    gt_prior = pm.Normal('gt_prior', mu=0, sigma=1, shape=gt_image.shape)
    
    # 将先验和PSF转换为Aesara张量
    gt_prior_tensor = gt_prior.reshape((1, 1, 10, 10))  # (batch_size, channels, height, width)
    psf_tensor = psf.reshape((1, 1, 10, 1))  # (output_channels, input_channels, filter_height, filter_width)
    
    # 卷积操作
    conv_op = at.nnet.conv2d(gt_prior_tensor, psf_tensor, border_mode='half', input_shape=(1, 1, 10, 10), filter_shape=(1, 1, 10, 1))
    
    # 卷积结果的形状调整
    conv_op = conv_op[0, 0]  # 去掉批次和通道维度
    
    # 定义似然函数
    likelihood = pm.Poisson('likelihood', mu=conv_op, observed=obs_image)
    
    # 采样
    trace = pm.sample(1000, return_inferencedata=True)

# 打印结果
print(az.summary(trace))