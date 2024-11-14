#接続、保存、学習、試験を一括(consolidation)で行う

#必要モジュールのimport
import socket
import time
import struct
import numpy as np
import winsound
import os

#torch関連の読み込み
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchsummary

#自作関数読み込み
import sys
sys.path.append('..')
from lib import readfile as rf
from lib import partsset
from learning.archive_nn import dataload as dl
#partsのセットを行う
from lib import partsset as ps

#必要パラメータ
motions={
    0:"suburi",
    1:"iai",
    2:"udehuri",
}

max_frames=20   #1データあたりのフレーム数
n_maxdata=15       #1モーションのデータ数
parts=27
dof=4




#加工用の仮置きでフォルダ作成
#timestampを手書きで

folder_path='processing/0612/'
if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
else:
    print("既にフォルダが埋まってます。\n別のフォルダ名にするか、既存フォルダを削除してください。\n")
    print(folder_path)
    exit()
for motion in motions:
    print("フォルダ名 "+folder_path)
    print(motions[motion])
    os.mkdir(folder_path+motions[motion])


host = ''
port = 52353


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((host, port))

# 受信用のバッファーサイズを設定
buffer_size = 4096




#motion_dataset=[data数][Nframe][27parts][4dof]
motion_dataset=np.empty((len(motions)*N,max_frames,parts,dof))

#学習のclass定義
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

model = MLP4(max_frames*parts*dof,4096,4096, len(motions))

#学習のための収録start
print(f"UDP 受信開始。ホスト: {host}, ポート: {port}")
n_files=0
fr=0
motion_id=0
print("3,2,1")
time.sleep(3)
print("start")
while True:
    if motion_id == len(motions):
        break
    try:
        # データを受信
        
        data, addr = udp_socket.recvfrom(buffer_size)
        #1575byteデータより大きいなら別データのためスキップ
        if(len(data)>1600):continue

        #１フレームの書き込み
        bnid_list = data.split(b'bnid')[1:]
        with open(+str(n_files)+'.txt', mode='a') as f:
            for id_parts,part in enumerate(bnid_list):
                tran_btdt_data = part.split(b'tran')[1:]
                dofdata = tran_btdt_data[0].split(b'btdt')[0]
                for id_dof,i in enumerate(range(0, 7*4, 4)):
                    float_value = struct.unpack('<f', dofdata[i:i+4])
                    f.write(f"{float_value[0]} ")
            f.write(f"\n")
            f.close()
        fr=fr+1
        if fr>150:
            fr=0
            n_files=n_files+1
            if n_files>n_maxdata:
                print(motions[motion_id],"が終わりました")
                n_files=0
                motion_id=motion_id+1
                if motion_id == len(motions):
                    print("学習のための収録が終わりました")
                    udp_socket.close()
                    break
                print(motions[motion_id],"を開始します")
                time.sleep(5)
            print(n_files,"番ファイルスタート")
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
        continue

#学習のための収録end

#学習を行うstart

#cudaの準備
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

#データ読み込み
#ファイル添え字の設定
Lnum_s=0
Lnum_e=n_maxdata-3
Lnum=Lnum_e-Lnum_s

Tnum_s=n_maxdata-3
Tnum_e=n_maxdata
Tnum=Tnum_e-Tnum_s

#frameの設定
fra_s=0
fra_e=120
fra_sep=20
fnum=int((fra_e-fra_s)/fra_sep)


parts_cut=0
parts_option_dict={
    0:ps.full_body,
    1:ps.upper_body,
    2:ps.lower_body,
    3:ps.left_arm,
    4:ps.right_arm,
    5:ps.left_leg,
    6:ps.right_leg,
}
fp=folder_path
labels_map=motions

#データロード開始
print("data load now!")
#--------
np_Ldata = dl.dataloading(fp,labels_map,Lnum_s,Lnum_e,fra_s,fra_e,fra_sep)
np_Ldata = parts_option_dict[parts_cut](np_Ldata)
#labelの添え字確認
np_Ldata_label=np.zeros((Lnum*fnum*len(labels_map)))
for i in range(len(labels_map)):
    np_Ldata_label[(i)*Lnum*fnum:(i+1)*Lnum*fnum] = i

t_data = torch.from_numpy(np_Ldata)
print(t_data.shape)
t_data_label = torch.from_numpy(np_Ldata_label)
print(t_data_label.shape)

#--------
np_Tdata = dl.dataloading(fp,labels_map,Tnum_s,Tnum_e,fra_s,fra_e,fra_sep)
np_Tdata = parts_option_dict[parts_cut](np_Tdata)
#labelの添え字確認
np_Tdata_label=np.zeros((Tnum*fnum*len(labels_map)))
for i in range(len(labels_map)):
    np_Tdata_label[(i)*Tnum*fnum:(i+1)*Tnum*fnum] = i

t_Tdata = torch.from_numpy(np_Tdata)
print(t_Tdata.shape)
t_Tdata_label = torch.from_numpy(np_Tdata_label)
print(t_Tdata_label.shape)
#-------

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


print(model)

# 損失関数（交差エントロピー）
loss_func = nn.CrossEntropyLoss(reduction='sum')

# パラメータ最適化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 学習の繰り返し回数
nepoch = 100

# 学習
results = []
print('# epoch  lossL  lossT  rateL  rateT')
for t in range(1, nepoch+1):
    lossL, rateL = train(model, loss_func, optimizer, dlL)
    lossT, rateT = evaluate(model, loss_func, dlT)
    results.append([t, lossL, lossT, rateL, rateT])
    if(t%10==0):
        print(f'{t}   {lossL:.5f}   {lossT:.5f}   {rateL:.4f}   {rateT:.4f}')


torch.save(model.state_dict(), model.save_name)
print('model saved')
#学習終了

#data=[Nframe][27parts][4dof]
in_data=np.empty((max_frames,parts,dof))

