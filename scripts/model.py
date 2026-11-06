import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 纯数学期望函数
def pure_expectation(k:np.ndarray, p:float) -> np.ndarray:
    """
    计算纯数学状态下的每人平均检测次数。
    :param k: 混检人数数组
    :param p: 真实阳性率
    :return: 对应的期望检测次数数组
    """
    q = 1 - p
    return 1 + (1 / k) - (q ** k)

# 解决绘图中中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 设置参数
N = 1293087   # 来源：邳州发布公众号（http://mp.weixin.qq.com/s?__biz=MjM5MDI5NTY4Nw==&mid=2650110537&idx=1&sn=ce52be08d7a9af3fac05d36cf7e1d7ad）
p = 3 / N     # 因本次筛查得出邳州市新冠阳性人数为 0 ，故使用三法则估计真实阳性率上界
k_values = np.arange(1, 31) # 假设 1 到 30 人混检，对比人均检测次数

y_values = pure_expectation(k_values, p)

# 绘图
plt.plot(k_values, y_values, marker='o', label='纯理论期望次数')
plt.axvline(x=10, color='red', linestyle='--', label='实际策略 k=10')
plt.xlabel('混检人数 k')
plt.ylabel('人均检测次数')
plt.title('探索性纯数学期望模型')
plt.legend()
plt.savefig(r"images\图1：探索性纯理论模型.png", dpi=300, bbox_inches='tight')
plt.show()