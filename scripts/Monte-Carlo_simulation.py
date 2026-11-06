import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def monte_carlo_simulation(N, p, k, trials=30):
    q = 1 - p
    num_groups = N // k
    total_tests_list = []
    
    for _ in range(trials):
        pop = np.random.binomial(1, p, N)
        
        groups = pop[:num_groups * k].reshape(num_groups, k)
        
        group_sums = groups.sum(axis=1)
        
        tests = np.where(group_sums == 0, 1, k + 1)
        
        total_tests_list.append(tests.sum())
        
    return np.mean(total_tests_list)

# 解决绘图中中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

N = 1293087   # 来源：邳州发布公众号（http://mp.weixin.qq.com/s?__biz=MjM5MDI5NTY4Nw==&mid=2650110537&idx=1&sn=ce52be08d7a9af3fac05d36cf7e1d7ad）
p = 3 / N     # 因本次筛查得出邳州市新冠阳性人数为 0 ，故使用三法则估计真实阳性率上界

print("正在运行百万级人群蒙特卡洛仿真...")
for k in [5, 10, 20, 30]:
    avg_tests = monte_carlo_simulation(N, p, k)
    print(f"当 k={k} 时，全城总检测次数约为: {avg_tests:.0f} 次（平均每人 {avg_tests/N:.4f} 次）")

k_range=np.arange(1,31)
means=[]
stds=[]
for k in k_range:
    res = monte_carlo_simulation(N, p, k, trials=30)
    means.append(np.mean(res))
    stds.append(np.std(res))

# 画图
plt.figure(figsize=(8, 5))
plt.plot(k_range, means, color='purple', marker='o', label='平均检测次数 (均值)')
# 画阴影代表波动范围 (均值 ± 1个标准差)
plt.fill_between(k_range, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                 color='purple', alpha=0.2, label='模拟波动范围 (±1 std)')

plt.axvline(x=10, color='red', linestyle='--', label='k=10')
plt.xlabel('混检人数 k')
plt.ylabel('全城总检测次数 (仿真)')
plt.title('蒙特卡洛仿真：检测次数随 k 的变化趋势')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('图5_蒙特卡洛带误差带.png', dpi=300, bbox_inches='tight')
plt.show()