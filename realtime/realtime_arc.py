import socket
import time
import struct
import numpy as np

#torch関連の読み込み
import torch
import torch.nn as nn
import torchsummary
import torch.nn.functional as F  # ソフトマックス用

host = ''
port = 5002


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#my home
#udp_socket.bind(("192.168.10.110", 5002))  # IPアドレスを指定してバインド

#ryukoku
udp_socket.bind(("133.83.82.105", 5002))

nframes=10
parts=6
dof=7

model_path='../nn/rawlearn.path'
fc_1=512
fc_2=512


choice_parts=[0,1,2]

#data=[Nframe][6parts][7dof]
in_data=np.empty((nframes,parts*dof))
choice_in_data=np.empty((nframes,len(choice_parts)*dof))
#最初のnframeまでは前側のデータが足りないため
#データがそろうまではmodel読み込みをスキップ
flag=0
minframe=50

motions=[
    #"freeze",
    "vslash2hand",
    "vslashleft",
    #"hslash2hand",
    "hslashleft",
    #"block",
    #"bowcharge",
    #"bowset",
    #"bowshot",
    "shake2hand",
    "shakeright",
    "shakeleft",
    #"lasso2hand",
    #"lassoright",
    #"lassoleft",
]


#学習した時のclass定義と揃える
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

model = MLP4(nframes*len(choice_parts)*dof,fc_1,fc_2, len(motions))
#モデルを読み込む
model.load_state_dict(torch.load(model_path))
print(f"UDP 受信開始。ホスト: {host}, ポート: {port}")
n=0
fr=0


for tm in range(3):
    print(3-tm)
    time.sleep(1)
print("start")

nframe=0
while True:
    try:
        data, addr = udp_socket.recvfrom(4096)  # データを受信
        

        #データがそろっていないときの処理
        if nframe<nframes:
            for i in range(6):
                tupledata=struct.unpack('<fffffff', data[i*40:i*40+28])
                for j in range(7):
                    in_data[nframe][i*7+j]=tupledata[j]
            j=0
            for i in choice_parts:
                choice_in_data[nframe][j*dof:(j+1)*dof]=in_data[nframe][i*dof:(i+1)*dof]
                j+=1
            nframe+=1
            continue
            
        #np_dataを更新して配列の最後frameを開ける
        #データをunpackする
        in_data[:-1] = in_data[1]
        for i in range(6):
            tupledata=struct.unpack('<fffffff', data[i*40:i*40+28])
            for j in range(7):
                in_data[-1][i*7+j]=tupledata[j]
        j=0
        for i in choice_parts:
            choice_in_data[-1][j*dof:(j+1)*dof]=in_data[-1][i*dof:(i+1)*dof]
            j+=1
        nframe+=1

        #nnに入力していく
        t_in_data = torch.from_numpy(choice_in_data).float()
        t_in_data = t_in_data.view(1, -1)
        Y = model(t_in_data)
        probs = F.softmax(Y, dim=1)  # ソフトマックスで確率に変換
        predicted_class = Y.argmax(dim=1).item()  # 最も確率の高いクラス
        confidence = probs[0, predicted_class].item()  # 確信度（確率）

        if confidence*100>75:
            print(f"予測クラス: {motions[predicted_class]} (確信度: {confidence * 100:.2f}%)")

                    
            
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
        continue