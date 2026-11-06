import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 解决绘图中中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 10合1灵敏度下降 11.11% (Watkins et al. 研究数据)，即 S(10) = 0.8889
lam = -np.log(0.8889) / (10 - 1) 

def comprehensive_cost(k, p, alpha=1000, beta=250):
    q = 1 - p
    # A: 基础检测成本
    test_cost = 1 + 1/k - q**k
    
    # B: 漏检风险
    S_k = np.exp(-lam * (k - 1))  # 灵敏度随k增加而下降
    risk_cost = alpha * (1 - S_k) * p 
    
    # C: 复检召回成本（一旦阳性，要召回k人）
    recall_cost = beta * (1 - q**k) * k
    
    return test_cost + risk_cost + recall_cost

# 计算综合实际代价
N = 1293087   # 来源：邳州发布公众号（http://mp.weixin.qq.com/s?__biz=MjM5MDI5NTY4Nw==&mid=2650110537&idx=1&sn=ce52be08d7a9af3fac05d36cf7e1d7ad）
p = 3 / N     # 因本次筛查得出邳州市新冠阳性人数为 0 ，故使用三法则估计真实阳性率上界
k_range = np.arange(2, 31)
costs = [comprehensive_cost(k, p) for k in k_range]

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(k_range, costs, marker='s', color='purple', label='综合实际代价（医学与管理约束）')
plt.axvline(x=10, color='red', linestyle='--', label='邳州实际策略 k=10')

# 找出最低点
min_idx = np.argmin(costs)
best_k = k_range[min_idx]
plt.scatter(best_k, costs[min_idx], color='gold', s=150, zorder=5, label=f'模型理论最优 k={best_k}')

plt.xlabel('混检人数 k')
plt.ylabel('综合代价（数值越小越优）')
plt.title('引入灵敏度与召回约束后的综合代价曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r'images\图3_综合代价曲线(beta=250).png', dpi=300, bbox_inches='tight')
plt.show()