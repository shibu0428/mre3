#nnflatの重ね合わせver

# 準備あれこれ
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
seaborn.set()
from torch.utils.data import Dataset

# PyTorch 関係のほげ
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
import torchsummary
import csv

#自作関数軍
import sys
sys.path.append('..')
from lib import readfile as rf
#partsのセットを行う
from lib import partsset as ps
#データをファイルから読み込むためのローダ
import learning.archive_nn.dataload as dl


#---------------------------------------------------
#パラメータここから
# クラス番号とクラス名
dataset_path="../dataset/"
dataset_days="1111"


positions={
    "sit",
    "sta",
}
position="sit"

motions=[
    "freeze",
    "vslash2hand",
    "vslashleft",
    "hslash2hand",
    "hslashleft",
    "block",
    "bowcharge",
    "bowset",
    "bowshot",
    "shake2hand",
    "shakeright",
    "shakeleft",
    "lasso2hand",
    "lassoright",
    "lassoleft",
]

model_save=1        #モデルを保存するかどうか 1なら保存
data_frames=2       #学習1dataあたりのフレーム数
all_data_frames=1800    #元データの読み取る最大フレーム数

choice_mode=0   #テストのチョイスを変更する
fc1=4096
fc2=8192
#パラメータここまで
#----------------------------------------------------------------------------------

data_cols=(7+2)*6       #csvの列数
cap_cols=7*6            #7DoFdataのみのデータの列数
data_n=int(all_data_frames/data_frames)  #切り分けた後のdataの数(重ね合わせなし)

learn_n=int(data_n*0.3)  #１モーションの学習のデータ数 3割を学習に
test_n=int(data_n-learn_n) #１モーションのテストのデータ数   7割をテストに
print(data_n,learn_n,test_n)


#cudaの準備
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())




#データロード開始
#all_data[モーションの種類][1モーションのframe][(7data+timestamp+deviceID)*6]->0.00123
all_data=np.zeros((len(motions),all_data_frames,data_cols))
all_cap_data=np.zeros((len(motions),all_data_frames,cap_cols))

for i in range(len(motions)):
    filepath=dataset_path+dataset_days+"_"+position+"_"+motions[i]+".csv"
    print(filepath)
    all_data[i]=np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames,:data_cols]
    all_cap_data[i]=np.delete(all_data[i],[7,8,16,17,25,26,34,35,43,44,52,53],1)
#all_dataからtimestampとdevIDを消去する->all_cap_data

np_data=np.zeros((learn_n*len(motions),data_frames,cap_cols))
np_data_label=np.zeros(learn_n*len(motions))
np_Tdata=np.zeros((test_n*len(motions),data_frames,cap_cols))
np_Tdata_label=np.zeros(test_n*len(motions))

print('np_data_shape=',np_data.shape)

for i in range(len(motions)):
    for f in range(learn_n):
        np_data[i*learn_n+f]=all_cap_data[i][f:f+data_frames]

for i in range(len(motions)):
    for f in range(test_n):
        np_Tdata[i*test_n+f]=all_cap_data[i][f+learn_n:f+data_frames+learn_n]

#ラベルセット
for i in range(len(motions)):
    np_data_label[learn_n*i:learn_n*(i+1)]=i
    np_Tdata_label[test_n*i:test_n*(i+1)]=i


#choiceモード(一部をテスト)は一時無効化
'''
if choice_mode==1:
    choice_test_n=data_n-int(all_data_frames/data_frames*0.3)
    np_choice_Tdata=np.zeros((choice_test_n*len(choice_test_motions),data_frames,data_cols))
    np_choice_Tdata_label=np.zeros(choice_test_n*len(choice_test_motions))
    print(choice_test_n,test_n,data_n,learn_n)
    for i in range(len(choice_test_motions)):
        choice_i=motions.index(choice_test_motions[i])
        np_choice_Tdata[i*test_n:(i+1)*test_n]=np_Tdata[choice_i*test_n:(choice_i+1)*test_n]
        np_choice_Tdata_label[i*choice_test_n:(i+1)*choice_test_n]=choice_i
'''



#numpy->torch
t_data = torch.from_numpy(np_data)
t_data_label = torch.from_numpy(np_data_label)
if choice_mode==0:
    t_Tdata = torch.from_numpy(np_Tdata)
    t_Tdata_label = torch.from_numpy(np_Tdata_label)
'''
else:
    t_Tdata = torch.from_numpy(np_choice_Tdata)
    t_Tdata_label = torch.from_numpy(np_choice_Tdata_label)
'''

class dataset_class(Dataset):
    def __init__(self,data,labels, transform=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)




# データ読み込みの仕組み
dsL = dataset_class(t_data,t_data_label)
dsT = dataset_class(t_Tdata,t_Tdata_label)
dlL = DataLoader(dsL, batch_size=10, shuffle=True)
dlT = DataLoader(dsT, batch_size=10, shuffle=False)
print(f'学習データ数: {len(dsL)}  テストデータ数: {len(dsT)}')


# 1epoch の学習を行う関数
#
def train(model, lossFunc, optimizer, dl):
    loss_sum = 0.0
    ncorrect = 0
    n = 0
    for i, (X, lab) in enumerate(dl):
        lab=lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()  # 入力データをFloat型に変換
        Y = model(X)           # 一つのバッチ X を入力して出力 Y を計算
        loss = lossFunc(Y, lab) # 正解ラベル lab に対する loss を計算
        optimizer.zero_grad()   # 勾配をリセット
        loss.backward()         # 誤差逆伝播でパラメータ更新量を計算
        optimizer.step()         # パラメータを更新
        n += len(X)
        loss_sum += loss.item()  # 損失関数の値
        ncorrect += (Y.argmax(dim=1) == lab).sum().item()  # 正解数

    return loss_sum/n, ncorrect/n

# 損失関数や識別率の値を求める関数
#
@torch.no_grad()
def evaluate(model, lossFunc, dl):
    loss_sum = 0.0
    ncorrect = 0
    n = 0
    for i, (X, lab) in enumerate(dl):
        lab=lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()  # 入力データをFloat型に変換
        Y = model(X)           # 一つのバッチ X を入力して出力 Y を計算
        loss = lossFunc(Y, lab)  # 正解ラベル lab に対する loss を計算
        n += len(X)
        loss_sum += loss.item() # 損失関数の値
        ncorrect += (Y.argmax(dim=1) == lab).sum().item()  # 正解数

    return loss_sum/n, ncorrect/n

##### 学習結果の表示用関数
# 学習曲線の表示
def printdata(m_size,subtitle):
  data = np.array(results)
  fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(12, 4))
  ax[0].plot(data[:, 0], data[:, 1], '.-', label='training data')
  ax[0].plot(data[:, 0], data[:, 2], '.-', label='test data')
  ax[0].axhline(0.0, color='gray')
  ax[0].set_ylim(-0.05, 1.75)
  ax[0].legend()
  ax[0].set_title(f'loss')
  ax[1].plot(data[:, 0], data[:, 3], '.-', label='training data')
  ax[1].plot(data[:, 0], data[:, 4], '.-', label='test data')
  ax[1].axhline(1.0, color='gray')
  ax[1].set_ylim(0.35, 1.01)
  ax[1].legend()
  ax[1].set_title(f'accuracy')
  fig.suptitle('modelSize'+str(m_size)+str(subtitle))

  # 学習後の損失と識別率
  loss2, rrate = evaluate(net, loss_func, dlL)
  print(f'# 学習データに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')
  loss2, rrate = evaluate(net, loss_func, dlT)
  print(f'# テストデータに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')
  plt.show()

class MLP4(nn.Module):

    # コンストラクタ． D: 入力次元数， H1, H2: 隠れ層ニューロン数， K: クラス数
    def __init__(self, D, H1, H2,K):
        super(MLP4, self).__init__()
        # 4次元テンソルで与えられる入力を2次元にする変換
        self.flatten = nn.Flatten()
        # 入力 => 隠れ層1
        self.fc1 = nn.Sequential(
            nn.Linear(D, H1), nn.Sigmoid()
        )
        # 隠れ層1から隠れ層2へ
        self.fc2 = nn.Sequential(
            nn.Linear(H1,H2), nn.Sigmoid()
        )

        # 隠れ層 => 出力層
        self.fc3 = nn.Linear(H2, K) # 出力層には活性化関数を指定しない


        # モデルの出力を計算するメソッド
    def forward(self, X):
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)

        return X


# ネットワークモデル

net = MLP4(data_frames*cap_cols,fc1,fc2, len(motions)).to(device)
#torchsummary.summary(net, (1, 28, 28))
print(net)

# 損失関数（交差エントロピー）
loss_func = nn.CrossEntropyLoss(reduction='sum')

# パラメータ最適化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

# 学習の繰り返し回数
nepoch = 200

# 学習
results = []
print('# epoch  lossL  lossT  rateL  rateT')
for t in range(1, nepoch+1):
    lossL, rateL = train(net, loss_func, optimizer, dlL)
    lossT, rateT = evaluate(net, loss_func, dlT)
    results.append([t, lossL, lossT, rateL, rateT])
    if(t%10==0):
        print(f'{t}   {lossL:.5f}   {lossT:.5f}   {rateL:.4f}   {rateT:.4f}')
printdata([fc1,fc2],"1111_15motions")
if model_save==0:
    exit(0)

torch.save(net.state_dict(),'rawlearn.path')
print('model saved')