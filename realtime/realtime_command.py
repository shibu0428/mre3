import socket
import time
import struct
import numpy as np

import pyautogui
import pydirectinput
import sys
import threading

#torch関連の読み込み
import torch
import torch.nn as nn
import torchsummary
import torch.nn.functional as F  # ソフトマックス用

#グラフ用
import matplotlib.pyplot as plt

host = ''
port = 5002


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#my home
udp_socket.bind(("192.168.10.110", 5002))  # IPアドレスを指定してバインド

#ryukoku
#udp_socket.bind(("133.83.82.105", 5002))

nframes=5
parts=6
dof=7

model_path='../nn/rawlearn.path'
fc_1=1024*2
fc_2=1024*2


choice_parts=[0,1,2,3,4,5]

#data=[Nframe][6parts][7dof]
in_data=np.empty((nframes,parts*dof))
choice_in_data=np.empty((nframes,len(choice_parts)*dof))
#最初のnframeまでは前側のデータが足りないため
#データがそろうまではmodel読み込みをスキップ
flag=0
minframe=50

motions=[
    "vslash",
    "hslash_underleft",
    "hslash_underright",
    "kick_leftleg",
    #"kick_rightleg",
    "walk_front",
    #"walk_right",
    #"walk_left",
    "tuki",
    "noutou_kosi",
    "noutou_senaka",
    #"freeze",
]

command_dict={
    "left_click":0,
    "2_click":1,
    "g":2,
    "left_Space":3,
    "right_Space":4,
    "w":5,
    "d":6,
    "a":7,
    "right_click":8,
    "g_Space":9,
    "Shift":10,
}



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

def command(command_name):
            case=command_name
            if case=="vslash":
                pydirectinput.click()
                
            elif case=="hslash_underleft":
                pydirectinput.click()
                
            elif case=="hslash_underright":
                pydirectinput.press('g')
                
            elif case=="kick_leftleg":
                pydirectinput.press('a')
                pydirectinput.press('Space')
                
            elif case=="kick_rightleg":
                pydirectinput.press('d')
                pydirectinput.press('Space')
                
            elif case=="walk_front":
                pydirectinput.press('w')
                
            elif case=="walk_right":
                pydirectinput.press('d')
                
            elif case=="walk_left":
                pydirectinput.press('a')
                
            elif case=="tuki":
                pydirectinput.click()
                
            elif case=="noutou_kosi":
                pydirectinput.click()
                
            elif case=="noutou_senaka":
                pydirectinput.press('Shift')


Y_hist=[]
probs_hist=[]
confidence_hist=[]

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
            nframe+=1
            continue
            
        #np_dataを更新して配列の最後frameを開ける
        #データをunpackする
        in_data = np.roll(in_data,-parts*dof)
        for i in range(6):
            tupledata=struct.unpack('<fffffff', data[i*40:i*40+28])
            for j in range(7):
                in_data[nframes-1][i*7+j]=tupledata[j]
        choice_in_data=in_data
        nframe+=1
        #for i in range(nframes):
            #print(in_data[i][0],end=',')
        #print()

        #nnに入力していく
        t_in_data = torch.from_numpy(choice_in_data).float()
        t_in_data = t_in_data.view(1, -1)
        Y = model(t_in_data)
        probs = F.softmax(Y, dim=1)  # ソフトマックスで確率に変換
        predicted_class = Y.argmax(dim=1).item()  # 最も確率の高いクラス
        confidence = probs[0, predicted_class].item()  # 確信度（確率）

        Y_hist.append(Y[0].detach().cpu().numpy())
        probs_hist.append(probs[0].detach().cpu().numpy())

        if confidence*100>75:
            print(f"予測クラス: {motions[predicted_class]} (確信度: {confidence * 100:.2f}%)")
            thread = threading.Thread(target=command, args=(motions[predicted_class],))
            thread.start()


            
        else:
            print(f"予測クラス: None (確信度: {confidence * 100:.2f}%)")

        if nframe>1000:
            break
                    
            
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
        continue



# UDPソケットを閉じる
udp_socket.close()

#ループ処理が終わったら各グラフを表示
g_title=input("Yの履歴グラフのタイトルを入力してください")

#Y_np_hist=np.array(probs_hist)
Y_np_hist=np.array(Y_hist)
# グラフをプロット
plt.figure()
for idx, motion in enumerate(motions):
    plt.plot(Y_np_hist[:, idx], label=motion)
plt.xlabel('frames')
plt.ylabel('predict')
plt.title(g_title)
plt.legend()
plt.show()

