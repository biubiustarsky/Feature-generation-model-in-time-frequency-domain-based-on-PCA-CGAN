import torch
import os
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as  np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
# 潜在的空间 其实GAN 训练出来的判别器对对抗样例的防御是有意义的。但是
from collections import defaultdict




src_dir = r'E:\kupyter\故障诊断\故障诊断\data\sim_feature'
nor_dir = os.path.join(src_dir, 'nor_feature.csv')
unb_dir = os.path.join(src_dir, 'unb_feature.csv')
mis_dir = os.path.join(src_dir, 'mis_feature.csv')
rub_dir = os.path.join(src_dir, 'rub_feature.csv')

nor = np.array(pd.read_csv(nor_dir))
unb = np.array(pd.read_csv(unb_dir))
mis = np.array(pd.read_csv(mis_dir))
rub = np.array(pd.read_csv(rub_dir))

# 归一化
# (X - Xmin) / (Xmax - Xmin)
# 去除orbit一列
for i in range(nor.shape[1] - 1):
    nor[:, i] = (nor[:, i] - min(nor[:, i])) / (max(nor[:, i]) - min(nor[:, i]))
    unb[:, i] = (unb[:, i] - min(unb[:, i])) / (max(unb[:, i]) - min(unb[:, i]))
    mis[:, i] = (mis[:, i] - min(mis[:, i])) / (max(mis[:, i]) - min(mis[:, i]))
    rub[:, i] = (rub[:, i] - min(rub[:, i])) / (max(rub[:, i]) - min(rub[:, i]))
nor_tsne=nor
unb_tsne=unb
mis_tsne=mis
rub_tsne=rub
labels_emd_dim=32


x1=nor_tsne[:,:-1]
a=torch.zeros(x1.shape[0],1)
x2=unb_tsne[:,:-1]
b=torch.ones(x2.shape[0],1)
x3=mis_tsne[:,:-1]
c=torch.ones(x3.shape[0],1)+torch.ones(x3.shape[0],1)
x4=rub_tsne[:,:-1]
d=torch.ones(x4.shape[0],1)*3
Y=np.concatenate((a, b, c,d), axis=0)# biaoqian
X=np.concatenate((x1, x2, x3, x4), axis=0)
import numpy as np
# 创建一维数组
a = np.array([1,2,3])
# 存储在当前目录
np.savetxt( "spambase.csv", a, delimiter="," )