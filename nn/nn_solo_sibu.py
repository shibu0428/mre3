# 必要なライブラリのインポート
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchsummary
import csv
from sklearn.metrics import f1_score

sns.set()

#---------------------------------------------------
# パラメータここから
dataset_path = "../dataset_sibu_solo/"
datasetdays = ["1", "2"]

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
    "golf",
    "turnR",
    #"turnL",
    "punchR",
    "punchL",
    "kickR",
    "kickL",
    "shakeR",
    "shakeL",
    #"walkleg",
    "jump",
    #"rhandR",
    #"rhandL",
]

name_list = [
    "sibu",
]

model_save = 0        # モデルを保存するかどうか 1なら保存
data_frames = 10      # 学習1dataあたりのフレーム数
all_data_frames = 560 + data_frames  # 元データの読み取る最大フレーム数

bs = 20   # バッチサイズ

fc1 = 1024 * 1
fc2 = 1024 * 1

# 学習の繰り返し回数
nepoch = 60

# ★変更: フレーム切り出しのステップ幅 (1なら従来通り)
frame_step = 1

# タイトルや保存ファイル名に step を含める
label_str='sprit '+str(data_frames)+'frames step:'+str(frame_step)+' unit:'+str(fc1)
save_path=str(data_frames)+'_'+str(frame_step)+'_'+str(fc1)+'solo_sibu2.png'

# ★変更: フレーム切り出しのステップ幅 (1なら従来通り)
frame_step = 1


choice_parts = [0, 1, 2, 3, 4, 5]
delete_parts = []

# ノイズの強さと生成回数
noise_level = 0.05
noise_repetitions = 0

# ===== 以下のフラグとリストを使って分割方法を制御します =====
split_by_date = True
split_by_person = False

train_names = ["sibu","haya","oga","gou"]
test_names  = ["yama"]

learn_par = 0.7
# パラメータここまで
#----------------------------------------------------------------------------------

data_cols = (7 + 2) * 6       # CSVの列数
cap_cols = 7 * 6              # 7DoFデータのみの列数

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

data_list = []
labels_list = []
person_labels_list = []
date_labels_list = []
date_to_index = {date: idx for idx, date in enumerate(datasetdays)}
name_to_index = {name: idx for idx, name in enumerate(name_list)}

# データを読み込み
for name in name_list:
    for date_idx, date in enumerate(datasetdays):
        for motion_idx, motion in enumerate(motions):
            filename = f"{motion}_{name}_{date}.csv"
            filepath = dataset_path + name + "/" + filename
            print(f"読み込み中: {filepath}")
            if not os.path.exists(filepath):
                print(f"警告: ファイルが存在しません -> {filepath}")
                continue

            data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]

            cap_data = np.delete(data, [7, 8, 16, 17, 25, 26, 34, 35, 43, 44, 52, 53], 1)

            cap_cols = len(choice_parts) * 7
            delete_list = []
            for i in delete_parts:
                delete_list.extend(range(i * 7, i * 7 + 7))
            cap_choice_data = np.delete(cap_data, delete_list, 1)

            data_list.append(cap_choice_data)
            labels_list.append(motion_idx)
            date_labels_list.append(date_idx)
            person_labels_list.append(name_to_index[name])

if split_by_date:
    train_dates = ["1"]
    test_dates = ["2"]
    train_date_indices = [date_to_index[date] for date in train_dates]
    test_date_indices = [date_to_index[date] for date in test_dates]

    train_data_sequences = []
    train_labels = []
    test_data_sequences = []
    test_labels = []

    for data_idx, data in enumerate(data_list):
        n_frames = data.shape[0]
        data_n = int(n_frames - data_frames)
        motion_label = labels_list[data_idx]
        date_label = date_labels_list[data_idx]

        if date_label in train_date_indices:
            # ★変更: frame_step で指定した刻み幅で切り出す
            for f in range(0, data_n, frame_step):
                sequence = data[f : f + data_frames]
                train_data_sequences.append(sequence)
                train_labels.append(motion_label)
        elif date_label in test_date_indices:
            for f in range(0, data_n, frame_step):
                sequence = data[f : f + data_frames]
                test_data_sequences.append(sequence)
                test_labels.append(motion_label)

    np_data = np.array(train_data_sequences)
    np_data_label = np.array(train_labels)
    np_Tdata = np.array(test_data_sequences)
    np_Tdata_label = np.array(test_labels)


def add_noise_multiple(data, noise_level=0.01, repetitions=1):
    augmented_data = [data]
    for _ in range(repetitions):
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_data.append(data + noise)
    return np.concatenate(augmented_data, axis=0)

np_data_augmented = add_noise_multiple(np_data, noise_level=noise_level, repetitions=noise_repetitions)
np_data_label_augmented = np.tile(np_data_label, noise_repetitions + 1)

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

dsL = dataset_class(t_data, t_data_label)
dsT = dataset_class(t_Tdata, t_Tdata_label)
dlL = DataLoader(dsL, batch_size=bs, shuffle=True)
dlT = DataLoader(dsT, batch_size=bs, shuffle=False)
print(f'学習データ数: {len(dsL)}  テストデータ数: {len(dsT)}')

def display_confusion_matrix(chart, class_names):
    df_cm = pd.DataFrame(chart.astype(int), index=class_names, columns=class_names)

    print("Confusion Matrix:")
    print(df_cm)
    plt.subplot(1, 3, 1)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    #plt.savefig(save_path)

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
    return chart.T

def printdata(m_size, subtitle):
    data = np.array(results)

    plt.subplot(1, 3, 2)
    plt.plot(data[:, 0], data[:, 1], '.-', label='training data')
    plt.plot(data[:, 0], data[:, 2], '.-', label='test data')
    plt.axhline(0.0, color='gray')
    plt.ylim(-0.05, 1.75)
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(data[:, 0], data[:, 3], '.-', label='training data')
    plt.plot(data[:, 0], data[:, 4], '.-', label='test data')
    plt.axhline(1.0, color='gray')
    plt.ylim(0.6, 1.01)
    plt.legend()
    plt.title('Accuracy')

    plt.suptitle('modelSize' + str(m_size) + str(subtitle))

    loss2, rrate = evaluate(net, loss_func, dlL)
    print(f'# 学習データに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')
    loss2, rrate = evaluate(net, loss_func, dlT)
    print(f'# テストデータに対する損失: {loss2:.5f}  識別率: {rrate:.4f}')

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

net = MLP4(data_frames * cap_cols, fc1, fc2, len(motions)).to(device)
print(net)

loss_func = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

results = []
print('# epoch  lossL  lossT  rateL  rateT')
for t in range(1, nepoch + 1):
    lossL, rateL = train(net, loss_func, optimizer, dlL)
    lossT, rateT = evaluate(net, loss_func, dlT)
    results.append([t, lossL, lossT, rateL, rateT])
    if (t % 1 == 0):
        print(f'{t:3d}   {lossL:.6f}   {lossT:.6f}   {rateL:.5f}   {rateT:.5f}')

chart = evaluate_test(net, dlT, len(motions))
print(chart)

plt.figure(figsize=(18, 5))
display_confusion_matrix(chart, motions)
printdata([fc1, fc2], label_str)
plt.tight_layout()
plt.savefig(save_path)
plt.show()

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

f1 = calculate_f1(net, dlT, len(motions))
print(f"F1 Score (weighted): {f1:.4f}")

if model_save == 0:
    exit(0)

torch.save(net.state_dict(), 'rawlearn.path')
print('model saved')
