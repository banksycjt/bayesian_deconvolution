import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# 自定义对数概率密度函数
def custom_logp(x, y):
    return -x**2 - y**2

def main():
    # 定义 PyMC 模型
    with pm.Model() as model:
        # 定义自定义分布
        x = pm.Uniform('x', lower=-10000, upper=10000)
        y = pm.Uniform('y', lower=-10000, upper=10000)
    
        # 定义联合对数概率密度
        pm.Potential('joint_logp', custom_logp(x, y))
    
        # 采样
        trace = pm.sample(1000, return_inferencedata=True)

    # 可视化结果
    az.plot_trace(trace)
    plt.show()

    # 绘制二维散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(trace.posterior['x'].values.flatten(), trace.posterior['y'].values.flatten(), alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Samples from exp(-x^2 - y^2)')
    plt.show()

if __name__ == '__main__':
    main()
    