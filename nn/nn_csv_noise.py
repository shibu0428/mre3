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
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchsummary
import csv

import pandas as pd
import seaborn as sns

from sklearn.metrics import f1_score

#---------------------------------------------------
# パラメータここから
dataset_path="../dataset/"
dataset_days="1121"

positions={
    "sit",
    "sta",
}
position="sit"

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

model_save=1        # モデルを保存するかどうか 1なら保存
data_frames=10      # 学習1dataあたりのフレーム数
all_data_frames=1800+data_frames  # 元データの読み取る最大フレーム数

bs=20   # バッチサイズ

fc1=512
fc2=512

# 学習の繰り返し回数
nepoch = 8

choice_parts=[0,1,2]
delete_parts=[3,4,5]

# パラメータ: ノイズの強さと生成回数を設定
noise_level = 2.5  # ノイズの強さ
noise_repetitions = 5  # ノイズ付きデータを生成する回数

# パラメータここまで
#----------------------------------------------------------------------------------

data_cols=(7+2)*6       # CSVの列数
cap_cols=7*6            # 7DoFデータのみの列数

learn_par=0.8

# CUDAの準備
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

# データロード開始
all_data=np.zeros((len(motions), all_data_frames, data_cols))
all_cap_data=np.zeros((len(motions), all_data_frames, cap_cols))

for i in range(len(motions)):
    filepath=dataset_path+dataset_days+"_"+position+"_"+motions[i]+".csv"
    print(filepath)
    all_data[i]=np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]
    all_cap_data[i]=np.delete(all_data[i], [7,8,16,17,25,26,34,35,43,44,52,53], 1)
# all_dataからtimestampとdevIDを消去する -> all_cap_data

# デバイス選択
cap_cols=len(choice_parts)*7
all_cap_choice_data=np.zeros((len(motions), all_data_frames, cap_cols))

delete_list=[]
for i in delete_parts:
    delete_list.extend(range(i*7, i*7+7))
print("delete cols = ", delete_list)

for i in range(len(motions)):
    all_cap_choice_data[i]=np.delete(all_cap_data[i], delete_list, 1)
if len(delete_parts) == 0:
    all_cap_choice_data=all_cap_data

# 重ね合わせあり
data_n=int(all_data_frames - data_frames)
learn_n=int(data_n * learn_par)
test_n=int(data_n - learn_n)

np_data=np.zeros((learn_n * len(motions), data_frames, cap_cols))
np_data_label=np.zeros(learn_n * len(motions))
np_Tdata=np.zeros((test_n * len(motions), data_frames, cap_cols))
np_Tdata_label=np.zeros(test_n * len(motions))

print('np_data_shape=', np_data.shape)

for i in range(len(motions)):
    for f in range(learn_n):
        np_data[i * learn_n + f]=all_cap_choice_data[i][f : f + data_frames]

for i in range(len(motions)):
    for f in range(test_n):
        np_Tdata[i * test_n + f]=all_cap_choice_data[i][f + learn_n : f + data_frames + learn_n]

# ラベルセット
for i in range(len(motions)):
    np_data_label[learn_n * i : learn_n * (i + 1)] = i
    np_Tdata_label[test_n * i : test_n * (i + 1)] = i

# === ここからノイズ付加を追加 ===
def add_noise_multiple(data, noise_level=0.01, repetitions=1):
    """
    ノイズを付加したデータを指定回数生成し、元データに結合する。
    :param data: 元データ (numpy array)
    :param noise_level: ノイズの強さ
    :param repetitions: ノイズ付きデータの生成回数
    :return: 元データとノイズ付きデータを結合した numpy array
    """
    augmented_data = [data]  # 元データを含むリスト
    for _ in range(repetitions):
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_data.append(data + noise)
    return np.concatenate(augmented_data, axis=0)





# ノイズを付加したデータを生成して、元データに結合
np_data_augmented = add_noise_multiple(np_data, noise_level=noise_level, repetitions=noise_repetitions)

# ノイズ付きデータのラベルは元データと同じものを繰り返す
np_data_label_augmented = np.tile(np_data_label, noise_repetitions + 1)


# numpy -> torch
t_data = torch.from_numpy(np_data_augmented)
t_data_label = torch.from_numpy(np_data_label_augmented)
t_Tdata = torch.from_numpy(np_Tdata)
t_Tdata_label = torch.from_numpy(np_Tdata_label)

class dataset_class(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# データ読み込みの仕組み
dsL = dataset_class(t_data, t_data_label)
dsT = dataset_class(t_Tdata, t_Tdata_label)
dlL = DataLoader(dsL, batch_size= bs, shuffle=True)
dlT = DataLoader(dsT, batch_size= bs, shuffle=False)
print(f'学習データ数: {len(dsL)}  テストデータ数: {len(dsT)}')

def display_confusion_matrix(chart, class_names):
    # pandas DataFrame に変換
    df_cm = pd.DataFrame(chart.astype(int), index=class_names, columns=class_names)

    # 表形式で表示
    print("Confusion Matrix:")
    print(df_cm)

    # ヒートマップの描画
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.show()


# 1epoch の学習を行う関数
def train(model, lossFunc, optimizer, dl):
    loss_sum = 0.0
    ncorrect = 0
    n = 0
    for i, (X, lab) in enumerate(dl):
        lab = lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()
        Y = model(X)
        loss = lossFunc(Y, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n += len(X)
        loss_sum += loss.item()
        ncorrect += (Y.argmax(dim=1) == lab).sum().item()
    return loss_sum / n, ncorrect / n

# 損失関数や識別率の値を求める関数
@torch.no_grad()
def evaluate(model, lossFunc, dl):
    loss_sum = 0.0
    ncorrect = 0
    n = 0
    for i, (X, lab) in enumerate(dl):
        lab = lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()
        Y = model(X)
        loss = lossFunc(Y, lab)
        n += len(X)
        loss_sum += loss.item()
        ncorrect += (Y.argmax(dim=1) == lab).sum().item()
    return loss_sum / n, ncorrect / n

def evaluate_test(model, dl, motions_len):
    chart = np.zeros([motions_len, motions_len])
    for j, (X, lab) in enumerate(dl):
        lab = lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()
        Y = model(X)
        for i in range(len(lab)):
            predicted = Y[i].argmax().item()
            actual = lab[i].item()
            chart[predicted][actual] += 1
    return chart

# 学習結果の表示用関数
def printdata(m_size, subtitle):
    data = np.array(results)
    fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(12, 4))
    ax[0].plot(data[:, 0], data[:, 1], '.-', label='training data')
    ax[0].plot(data[:, 0], data[:, 2], '.-', label='test data')
    ax[0].axhline(0.0, color='gray')
    ax[0].set_ylim(-0.05, 1.75)
    ax[0].legend()
    ax[0].set_title('loss')
    ax[1].plot(data[:, 0], data[:, 3], '.-', label='training data')
    ax[1].plot(data[:, 0], data[:, 4], '.-', label='test data')
    ax[1].axhline(1.0, color='gray')
    ax[1].set_ylim(0.35, 1.01)
    ax[1].legend()
    ax[1].set_title('accuracy')
    fig.suptitle('modelSize' + str(m_size) + str(subtitle))
    # 学習後の損失と識別率
    loss2, rrate = evaluate(net, loss_func, dlL)
    print(f'# 学習データに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')
    loss2, rrate = evaluate(net, loss_func, dlT)
    print(f'# テストデータに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')
    plt.show()

class MLP4(nn.Module):
    def __init__(self, D, H1, H2, K):
        super(MLP4, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(D, H1), nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(H1, H2), nn.Sigmoid()
        )
        self.fc3 = nn.Linear(H2, K)
    def forward(self, X):
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


# ネットワークモデル
net = MLP4(data_frames * cap_cols, fc1, fc2, len(motions)).to(device)
print(net)

# 損失関数（交差エントロピー）
loss_func = nn.CrossEntropyLoss(reduction='sum')

# パラメータ最適化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

# 学習
results = []
print('# epoch  lossL  lossT  rateL  rateT')
for t in range(1, nepoch + 1):
    lossL, rateL = train(net, loss_func, optimizer, dlL)
    lossT, rateT = evaluate(net, loss_func, dlT)
    results.append([t, lossL, lossT, rateL, rateT])
    if (t % 10 == 0):
        print(f'{t:3d}   {lossL:.6f}   {lossT:.6f}   {rateL:.5f}   {rateT:.5f}')

chart = evaluate_test(net, dlT, len(motions))
display_confusion_matrix(chart, motions)
printdata([fc1, fc2], "1111_15motions")

# F1スコアを計算する関数
def calculate_f1(model, dl, motions_len):
    all_true_labels = []
    all_pred_labels = []
    for X, lab in dl:
        lab = lab.long()
        X, lab = X.to(device), lab.to(device)
        X = X.float()
        Y = model(X)
        preds = torch.argmax(Y, dim=1)
        all_true_labels.extend(lab.cpu().numpy())
        all_pred_labels.extend(preds.cpu().numpy())
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    return f1

# テストデータに対するF1スコアを計算して表示
f1 = calculate_f1(net, dlT, len(motions))
print(f"F1 Score (weighted): {f1:.4f}")

if model_save == 0:
    exit(0)

torch.save(net.state_dict(), 'rawlearn.path')
print('model saved')
