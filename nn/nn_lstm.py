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
import torchsummary
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns

#---------------------------------------------------
# パラメータここから
dataset_path = "../dataset/"
datasetdays = ["1111","1121","1128"]

position = "sit"
motions = [
    "vslash2hand",
    "vslashleft",
    "hslashleft",
]

model_save = 1        # モデルを保存するかどうか 1なら保存
data_frames = 10      # LSTM用にシーケンス長を調整
all_data_frames = 1800 + data_frames

bs = 20   # バッチサイズ
hidden_dim = 128  # LSTM隠れ層の次元数
num_layers = 1    # LSTMの層数

# 学習の繰り返し回数
nepoch = 20

choice_parts = [0, 1, 2]
delete_parts = [3, 4, 5]

# パラメータ: ノイズの強さと生成回数を設定
noise_level = 0.01
noise_repetitions = 1

split_by_date = False  # 日付でデータを分割するかどうか

# パラメータここまで
#---------------------------------------------------

data_cols = (7 + 2) * 6       # CSVの列数
cap_cols = 7 * 6              # 7DoFデータのみの列数
learn_par = 0.8

# CUDAの準備
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# データロード開始
data_list = []
labels_list = []
date_labels_list = []

for date in datasetdays:
    for motion_idx, motion in enumerate(motions):
        filepath = f"{dataset_path}{date}_{position}_{motion}.csv"
        print(filepath)
        data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]
        cap_data = np.delete(data, [7, 8, 16, 17, 25, 26, 34, 35, 43, 44, 52, 53], 1)
        delete_list = []
        for i in delete_parts:
            delete_list.extend(range(i * 7, i * 7 + 7))
        cap_choice_data = np.delete(cap_data, delete_list, 1)
        data_list.append(cap_choice_data)
        labels_list.append(motion_idx)

# データ分割と前処理
data_sequences_list = []
data_labels_list = []

for data_idx, data in enumerate(data_list):
    n_frames = data.shape[0]
    data_n = n_frames - data_frames
    motion_label = labels_list[data_idx]

    for f in range(data_n):
        sequence = data[f:f + data_frames]
        data_sequences_list.append(sequence)
        data_labels_list.append(motion_label)

data_sequences_array = np.array(data_sequences_list)
data_labels_array = np.array(data_labels_list)

np_data, np_Tdata, np_data_label, np_Tdata_label = train_test_split(
    data_sequences_array, data_labels_array, test_size=1 - learn_par, shuffle=True, stratify=data_labels_array
)

# ノイズ追加関数
def add_noise(data, noise_level=0.01, repetitions=1):
    augmented_data = [data]
    for _ in range(repetitions):
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_data.append(data + noise)
    return np.concatenate(augmented_data, axis=0)

np_data_augmented = add_noise(np_data, noise_level=noise_level, repetitions=noise_repetitions)
np_data_label_augmented = np.tile(np_data_label, noise_repetitions + 1)

# numpy -> torch
t_data = torch.from_numpy(np_data_augmented).float()
t_data_label = torch.from_numpy(np_data_label_augmented).long()
t_Tdata = torch.from_numpy(np_Tdata).float()
t_Tdata_label = torch.from_numpy(np_Tdata_label).long()

# DatasetとDataLoaderの準備
class DatasetLSTM(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

dsL = DatasetLSTM(t_data, t_data_label)
dsT = DatasetLSTM(t_Tdata, t_Tdata_label)
dlL = DataLoader(dsL, batch_size=bs, shuffle=True)
dlT = DataLoader(dsT, batch_size=bs, shuffle=False)

# --- LSTMモデル定義 ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        lstm_out, (hn, cn) = self.lstm(X)
        out = self.fc(hn[-1])
        return out

input_dim = t_data.shape[2]  # 各時刻の特徴量次元
output_dim = len(motions)  # クラス数
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)

# 損失関数と最適化
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 学習関数
def train(model, loss_func, optimizer, dl):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for X, y in dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_correct += (output.argmax(dim=1) == y).sum().item()
    return total_loss / len(dl.dataset), total_correct / len(dl.dataset)

# 評価関数
def evaluate(model, loss_func, dl):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_func(output, y)
            total_loss += loss.item() * X.size(0)
            total_correct += (output.argmax(dim=1) == y).sum().item()
    return total_loss / len(dl.dataset), total_correct / len(dl.dataset)

# 学習ループ
results = []
for epoch in range(1, nepoch + 1):
    train_loss, train_acc = train(model, loss_func, optimizer, dlL)
    val_loss, val_acc = evaluate(model, loss_func, dlT)
    results.append((epoch, train_loss, val_loss, train_acc, val_acc))
    print(f"Epoch {epoch}/{nepoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# モデルの保存
if model_save:
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved.")

# 結果のプロット
epochs, train_losses, val_losses, train_accs, val_accs = zip(*results)
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()
