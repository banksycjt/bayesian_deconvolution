import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def generate_siemens_star(size, n_spokes):
    """生成 Siemens Star 图像"""
    star = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            angle = np.arctan2(i - center, j - center)
            spoke = int((angle / np.pi + 1) * n_spokes / 2) % n_spokes
            star[i, j] = 1 if spoke % 2 == 0 else 0
    return star

def gaussian_psf(size, sigma):
    """生成高斯点扩散函数"""
    x = np.arange(-size // 2, size // 2)
    y = np.arange(-size // 2, size // 2)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf

def generate_observed_image(I_true, psf):
    """生成观测图像（卷积 + 泊松噪声）"""
    I_convolved = convolve2d(I_true, psf, mode='same', boundary='wrap')
    I_observed = np.random.poisson(I_convolved)
    return I_observed

def estimate_ground_truth(I_observed, psf, size, n_spokes, n_samples=1000, tune=1000):
    """使用 PyMC 估计 ground truth 图像"""
    with pm.Model() as model:
        # 定义 ground truth 图像的先验分布
        I_true_prior = pm.Uniform('I_true_prior', lower=0, upper=1, shape=(size, size))
        
        # 卷积操作
        I_convolved_model = pm.math.conv(I_true_prior, psf, mode='full')[:, :size, :size]
        
        # 泊松噪声模型
        I_observed_model = pm.Poisson('I_observed', mu=I_convolved_model, observed=I_observed)
        
        # 采样
        trace = pm.sample(n_samples, tune=tune, cores=1)
    
    return trace

def plot_results(I_true, I_observed, trace):
    """可视化结果"""
    # 绘制 ground truth 图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(I_true, cmap='gray')
    plt.title('Ground Truth')
    
    # 绘制观测图像
    plt.subplot(1, 3, 2)
    plt.imshow(I_observed, cmap='gray')
    plt.title('Observed Image')
    
    # 绘制后验均值图像
    plt.subplot(1, 3, 3)
    I_true_posterior_mean = trace['I_true_prior'].mean(axis=0)
    plt.imshow(I_true_posterior_mean, cmap='gray')
    plt.title('Posterior Mean')
    
    plt.tight_layout()
    plt.show()

def main():
    # 参数设置
    size = 128
    n_spokes = 16
    sigma = 2.0
    psf_size = 15
    n_samples = 1000
    tune = 1000
    
    # 生成 Siemens Star 图像
    I_true = generate_siemens_star(size, n_spokes)
    
    # 生成高斯点扩散函数
    psf = gaussian_psf(psf_size, sigma)
    
    # 生成观测图像
    I_observed = generate_observed_image(I_true, psf)
    
    # 使用 PyMC 估计 ground truth 图像
    trace = estimate_ground_truth(I_observed, psf, size, n_spokes, n_samples, tune)
    
    # 可视化结果
    plot_results(I_true, I_observed, trace)

if __name__ == '__main__':
    main()