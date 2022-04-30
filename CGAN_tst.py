
import os
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as  np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
# 潜在的空间 其实GAN 训练出来的判别器对对抗样例的防御是有意义的。但是
from collections import defaultdict
import torch.utils.data as Data
import pandas as pd  # 这个包用来读取CSV数据
import torch
from torch.nn import functional as F


# 继承Dataset，定义自己的数据集类 mydataset
class mydataset(Data.Dataset):
    def __init__(self, csv_file):   # self 参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        data_csv = pd.DataFrame(pd.read_csv(csv_file))   # 读数据
        # self.csv_data = data_csv.drop(axis=1, columns='58', inplace=False)  # 删除最后一列标签
    def __len__(self):
        return len(self.csv_data)
    def __getitem__(self, idx):
        data = self.csv_data.values[idx]
        return data



src_dir = r'E:\kupyter\故障诊断\故障诊断\data\sim_feature'
nor_dir = os.path.join(src_dir, 'nor_feature.csv')
unb_dir = os.path.join(src_dir, 'unb_feature.csv')
mis_dir = os.path.join(src_dir, 'mis_feature.csv')
rub_dir = os.path.join(src_dir, 'rub_feature.csv')

nor = np.array(pd.read_csv(nor_dir))
unb = np.array(pd.read_csv(unb_dir))
mis = np.array(pd.read_csv(mis_dir))
rub = np.array(pd.read_csv(rub_dir))


for i in range(nor.shape[1] - 1):
    nor[:, i] = (nor[:, i] - min(nor[:, i])) / (max(nor[:, i]) - min(nor[:, i]))
    unb[:, i] = (unb[:, i] - min(unb[:, i])) / (max(unb[:, i]) - min(unb[:, i]))
    mis[:, i] = (mis[:, i] - min(mis[:, i])) / (max(mis[:, i]) - min(mis[:, i]))
    rub[:, i] = (rub[:, i] - min(rub[:, i])) / (max(rub[:, i]) - min(rub[:, i]))
nor_tsne=nor
unb_tsne=unb
mis_tsne=mis
rub_tsne=rub



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



data = mydataset('spambase.csv')
X_pca=PCA(n_components=0.90)
X_pca.fit(X)
X_pca=X_pca.transform(X)
X = torch.tensor(X_pca).long()
Y = torch.tensor(Y).long()
torch_dataset = Data.TensorDataset(X, Y)  # 对给定的 tensor 数据，将他们包装成 dataset

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.hidden = nn.Linear(100+4, 256)  # 隐藏层
        self.out = nn.Linear(256, X.shape[1])  # 输出层


    def forward(self, z):

        return self.out(F.relu(self.hidden(z)))




class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.hidden = nn.Linear(8, 256)  # 隐藏层
        self.out = nn.Linear(256, 1)  # 输出层


    def forward(self,input):


        return torch.sigmoid(self.out(F.relu(self.hidden(input))))

#
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['g_loss'], label='g loss')
    ax1.plot(history['d_loss'], label='d loss')

    ax1.set_ylim([0, 2])
    ax1.legend()
    ax1.set_ylabel('D_G_Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(history['fake_loss'], label='fake loss')
    ax2.plot(history['real_loss'], label='real loss')

    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.set_ylabel('fake_loss')
    ax2.set_xlabel('Epoch')

    fig.suptitle('Training History')
    plt.show()


batch_size=50
num_epoch=5000
generator = Generator()
discriminator = Discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))#, weight_decay=0.0001
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))#, weight_decay=0.0001

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)


history = defaultdict(list)  # 构建一个默认value为list的字典

data_loader = DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(data_loader):

        gt_data,labels = mini_batch
        z = torch.randn(batch_size,100)
        labels=labels.reshape(batch_size)
        # print(labels.shape)
        embd = nn.Embedding(100, 4)(labels)
        # embd=nn.Embedding(100, 4)(labels).t()
        # print(embd.shape)
        # print(z.shape)
        z=torch.cat((z, embd),1)

        # print(z.shape)

        pred_data = generator(z)
        g_optimizer.zero_grad()
        recons_loss = torch.abs(pred_data - gt_data).mean()
        pred_data_all=torch.cat((pred_data, embd), 1)
        gt_data_all = torch.cat((gt_data, embd), 1)

        g_loss = recons_loss * 0.05 + loss_fn(discriminator(pred_data_all), labels_one)
        g_loss.backward()
        g_optimizer.step()
        d_optimizer.zero_grad()

        real_loss = loss_fn(discriminator(gt_data_all.detach()), labels_one)
        fake_loss = loss_fn(discriminator(pred_data_all.detach()), labels_zero)

        d_loss = (real_loss + fake_loss)
        d_loss.backward()
        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了
        d_optimizer.step()

        if epoch % 1== 0:
            print(
                # f"step:{len(data_loader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
            #  netD.zero_grad()
                f"step:{epoch}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
            history['g_loss'].append(g_loss.item())
            history['d_loss'].append(d_loss.item())
            history['fake_loss'].append(fake_loss.item())
            history['real_loss'].append(real_loss.item())
            if (epoch+1)% num_epoch==0:
                last_data = pred_data.detach().numpy()
                print(last_data.shape)
                datax = pd.DataFrame(last_data)
                datax.to_csv('lastdata.csv', header=False, index=False)



plot_training_history(history)



