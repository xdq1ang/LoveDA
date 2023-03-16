import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

path = r"icsc\GF2_PMS1__L1A0001037899-MSS1_20.npy"
save_path = path.replace("npy", "png")
prob = np.load(path)

# 取最大值概率 
prob_1stMax = np.max(prob, axis=0)
# 取第二概率
prob_2stMax = np.sort(prob, axis=0)[::-1][1,:,:]

# 求绝对差异图
error = prob_1stMax - prob_2stMax

sns.heatmap(data=error)
# plt.show()
plt.savefig(save_path)


