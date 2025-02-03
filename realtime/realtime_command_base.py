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
#udp_socket.bind(("192.168.10.110", 5002))  # IPアドレスを指定してバインド

#ryukoku
udp_socket.bind(("133.83.82.105", 5002))

nframes=10
parts=6
dof=7

model_path='../fine_models/sibu2_fine.path'
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

motions = [
    "vslash",
    "hslash_ul",
    "hslash_ur",
    "thrust",
    "noutou_koshi",
    "noutou_senaka",
    "roll_r",
    "roll_l",
    "walk",
]

scaling_flag=0  #1ならスケーリングをおこなう
scale_par=np.array([
    0.4,#vslash
    0.4,#hslash_ul
    0.4,#hslash_ur
    0.4,#thrust
    0.4,#noutou_kosi
    0.4,#noutou_senaka
    0.4,#roll_r
<<<<<<< HEAD
    0.3,#roll_l
    0.3,#walk
=======
    0.4,#roll_l
    0.4,#walk

>>>>>>> bcb4da6de39e27430a43b81ec9547959751cc3cf
])

baseline_probs = np.zeros(len(motions))
BASELINE_FRAMES = 100  # 何フレーム分をベースラインとして集計するか
baseline_count = 0
baseline_acquired = False


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


for tm in range(5):
    print(3-tm)
    time.sleep(1)
print("start")

def command(command_name):
            case=command_name
            if case=="vslash":
                pydirectinput.press('v')
                
            elif case=="hslash_ul":
                pydirectinput.press('u')
                
            elif case=="hslash_ur":
                pydirectinput.press('b')
                
            elif case=="thrust":
                pydirectinput.press('h')
                
            elif case=="roll_l":
                pydirectinput.keyDown('a')
                pydirectinput.press('r')
                pydirectinput.keyUp('a')
                
            elif case=="roll_r":
                pydirectinput.keyDown('d')
                pydirectinput.press('r')
                pydirectinput.keyUp('d')
                
            elif case=="walk":
                pydirectinput.press('w')
                
            elif case=="noutou_kosi":
                pydirectinput.press('p')
                
            elif case=="noutou_senaka":
                pydirectinput.press('n')


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

        #ここからベースライン決定
        probs_array = probs[0].detach().cpu().numpy()  # numpy配列化
        if not baseline_acquired:
            baseline_probs += probs_array
            baseline_count += 1
    
            # ある程度のフレーム数を集めたら平均ベースラインを確定
            if baseline_count >= BASELINE_FRAMES:
                baseline_probs /= baseline_count
                baseline_probs=baseline_probs*scale_par
                baseline_acquired = True
                print("Baseline acquired.")
    
            # ベースライン取得が完了するまではコマンド出力しない
            continue

        if scaling_flag==1:
            corrected_probs = probs_array - baseline_probs
        else:
            corrected_probs = probs_array
        corrected_probs = np.clip(corrected_probs, 0.0, None)

        '''
        # 正規化して合計を1に
        sum_corrected = corrected_probs.sum()
        if sum_corrected > 0:
            corrected_probs /= sum_corrected
        '''
        predicted_class = np.argmax(corrected_probs)
        confidence = corrected_probs[predicted_class]
        '''
        predicted_class = Y.argmax(dim=1).item()  # 最も確率の高いクラス
        confidence = probs[0, predicted_class].item()  # 確信度（確率）
        '''
        Y_hist.append(Y[0].detach().cpu().numpy())
        probs_hist.append(corrected_probs)

        if confidence*100>85:
            print(f"予測クラス: {motions[predicted_class]} (確信度: {confidence * 100:.2f}%)")
            thread = threading.Thread(target=command, args=(motions[predicted_class],))
            thread.start()


            
        else:
            print(f"予測クラス: None (確信度: {confidence * 100:.2f}%)")

        if nframe>2000:
            break
                    
            
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
        continue


print(baseline_probs)
# UDPソケットを閉じる
udp_socket.close()

#ループ処理が終わったら各グラフを表示
g_title=input("Yの履歴グラフのタイトルを入力してください")

Y_np_hist=np.array(probs_hist)
#Y_np_hist=np.array(Y_hist)
# グラフをプロット
plt.figure()
for idx, motion in enumerate(motions):
    plt.plot(Y_np_hist[:, idx], label=motion)
plt.xlabel('frames')
plt.ylabel('predict')
plt.title(g_title)
plt.legend()
plt.show()

