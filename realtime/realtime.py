import socket
import time
import struct
import numpy as np

# torch関連の読み込み
import torch
import torch.nn as nn
import torch.nn.functional as F  # ソフトマックス用

#グラフ用
import matplotlib.pyplot as plt

# UDP設定
host = ''
port = 5002
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ホスト設定 (ローカル環境またはリモート環境に応じて設定)
# udp_socket.bind(("192.168.10.110", 5002))  # My home
udp_socket.bind(("133.83.82.105", 5002))  # Ryukoku

# 定数設定
nframes = 10  # フレーム数
parts = 6  # 部位の数
dof = 7  # Degree of Freedom (自由度)
model_path = '../nn/rawlearn.path'
fc_1, fc_2 = 512, 512
choice_parts = [0, 1, 2]  # 選択する部位のインデックス
confidence_threshold = 75  # 確信度の閾値 (%)
temperature = 1.2  # 温度スケーリングのパラメータ

# モーションラベル (学習時のクラス定義に揃える)
motions=[
    #"freeze",
    "vslash2hand",
    "vslashleft",
    #"hslash2hand",
    "hslashleft",
    #"lasso2hand",
    "lassoright",
    "lassoleft",
    "walkslow",
    "walkfast",
    #a
    # "clap",
]

# データバッファを初期化
in_data = np.empty((nframes, parts * dof))
choice_in_data = np.empty((nframes, len(choice_parts) * dof))

# NNモデル定義 (学習時と同じ構造)
class MLP4(nn.Module):
    def __init__(self, D, H1, H2, K):
        super(MLP4, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(D, H1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(H1, H2),
            nn.Sigmoid()
        )
        self.fc3 = nn.Linear(H2, K)

    def forward(self, X):
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X

# モデルの読み込み
model = MLP4(nframes * len(choice_parts) * dof, fc_1, fc_2, len(motions))
model.load_state_dict(torch.load(model_path))
print(f"UDP 受信開始。ホスト: {host}, ポート: {port}")

# 処理開始前のカウントダウン
for tm in range(3, 0, -1):
    print(tm)
    time.sleep(1)
print("start")

# メインループ
nframe = 0
freeze_flag=0   #閾値以下を識別した時に連続した場合のフラグ
freeze_time=0   #flagがonの時のフレーム回数を表示

Y_hist=[]
probs_hist=[]
confidence_hist=[]

while True:
    try:
        # データを受信
        data, addr = udp_socket.recvfrom(4096)

        # 初期フレームが揃っていない場合
        if nframe < nframes:
            for i in range(parts):
                tupledata = struct.unpack('<fffffff', data[i * 40:i * 40 + 28])
                for j in range(dof):
                    in_data[nframe, i * dof + j] = tupledata[j]
            for j, part in enumerate(choice_parts):
                choice_in_data[nframe, j * dof:(j + 1) * dof] = in_data[nframe, part * dof:(part + 1) * dof]
            nframe += 1
            continue

        # データをシフトして最新フレームを格納
        in_data[:-1] = in_data[1:]
        for i in range(parts):
            tupledata = struct.unpack('<fffffff', data[i * 40:i * 40 + 28])
            in_data[-1, i * dof:(i + 1) * dof] = tupledata

        # 選択した部位データの更新
        for j, part in enumerate(choice_parts):
            choice_in_data[-1, j * dof:(j + 1) * dof] = in_data[-1, part * dof:(part + 1) * dof]

        # モデル入力の整形
        t_in_data = torch.from_numpy(choice_in_data).float().view(1, -1)

        # 推論
        Y = model(t_in_data)
        probs = F.softmax(Y / temperature, dim=1)  # 温度スケーリング適用
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item() * 100  # 確率をパーセント表記
        
        Y_hist.append(Y[0].detach().cpu().numpy())
        Y_hist.append(Y[0].detach().cpu().numpy())

        # 確信度が閾値を超えた場合のみ出力
        if confidence > confidence_threshold:
            print(f"予測クラス: {motions[predicted_class]} (確信度: {confidence:.2f}%)")
            freeze_flag=0
            freeze_time=0
        elif freeze_flag == 0:   #freezeが連続ではない
            freeze_flag = 1
            freeze_time = 1
            print(print('\rfreeze frame=%d' %freeze_time, end=''))
        else:
            freeze_time+=1
            print(f'\r予測クラス: {motions[predicted_class]} (確信度: {confidence:.2f}%)freeze frame={freeze_time}', end='')
            if freeze_time>400:
                break
        


    except OSError as e:
        # エラーが発生した場合
        print(f"エラー: {e}")
        continue
# UDPソケットを閉じる
udp_socket.close()

#ループ処理が終わったら各グラフを表示
g_title=input("Yの履歴グラフのタイトルを入力してください")

Y_np_hist=np.array(Y_hist)
# グラフをプロット
plt.figure()
for idx, motion in enumerate(motions):
    plt.plot(Y_np_hist[:, idx], label=motion)
plt.xlabel('フレーム数')
plt.ylabel('モデルの出力')
plt.title(g_title)
plt.legend()
plt.show()

g_title=input("スケーリング後のグラフのタイトルを入力してください")

probs_np_hist=np.array(probs_hist)
# グラフをプロット
plt.figure()
for idx, motion in enumerate(motions):
    plt.plot(probs_np_hist[:, idx], label=motion)
plt.xlabel('フレーム数')
plt.ylabel('モデルの出力')
plt.title(g_title)
plt.legend()
plt.show()
