import socket
import time
import struct
import numpy as np

#torch関連の読み込み
import torch
import torch.nn as nn
import torchsummary


host = ''
port = 52353


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((host, port))

# 受信用のバッファーサイズを設定
buffer_size = 8192

nframes=6
parts=6
dof=7

model_path=''
fc_1=2048
fc_2=4096


#data=[Nframe][6parts][7dof]
in_data=np.empty((nframes,parts,dof))

#最初のnframeまでは前側のデータが足りないため
#データがそろうまではmodel読み込みをスキップ
flag=0
minframe=50

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

model = MLP4(nframes*parts*dof,fc_1,fc_2, len(motions))
#モデルを読み込む
model.load_state_dict(torch.load(model_path))
print(f"UDP 受信開始。ホスト: {host}, ポート: {port}")
outfile=input("output file name?")
n=0
fr=0

for tm in range(3):
    print(3-tm)
    time.sleep(1)
print("start")

nframe=0
while True:
    try:
        data, addr = udp_socket.recvfrom(1024)  # データを受信
        nframe+=1

        #開始のフレームより早いなら処理しない
        if nframe<minframe:
            continue


                    
            
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
        continue
