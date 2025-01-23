import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

# --------------------------------------------------------------
# 1. 「再学習用データ」と「テスト用データ」を指定
# --------------------------------------------------------------

# データセットが格納されているパス
dataset_path = "../dataset_name/"

# 再学習に使う人物と日付
fine_tuning_person = "gou"   # 例："gou"
fine_tuning_days = ["1"]     # 例：3日目だけ使う

# テストに使う人物と日付（再学習には使わない）
test_person = "gou"          # 同じ人物の別日、あるいは別人を指定してもよい
test_days = ["2","3","4","5"]            # 例：4日目だけ使う

# モーションの種類
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
    #"golf"
]


fine_tune_epochs = 10  # 例: 10エポック
# 取り出すフレーム数など
data_frames = 10
all_data_frames = 580 + data_frames
data_cols = (7 + 2) * 6

# 選択するパーツ (各パーツ7DoF)
choice_parts = [0, 1, 2, 3, 4, 5]
delete_parts = []  # 削除したいパーツがあればリストに入れる

fc1 = 1024 * 2
fc2 = 1024 * 2

# バッチサイズ
batch_size = 20

# CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# --------------------------------------------------------------
# 2. データ読み込み用の関数
# --------------------------------------------------------------
def load_person_data(
    person, days, dataset_path, motions,
    data_frames=5, all_data_frames=600,
    data_cols=54, choice_parts=None, delete_parts=None
):
    """
    特定の person と複数日 days からモーションCSVを読み込み、
    data_frames個ずつ切り出したシーケンスとラベルを返す。
    """
    sequences = []
    labels = []

    for date in days:
        for motion_idx, motion in enumerate(motions):
            filename = f"{motion}_{person}_{date}.csv"
            filepath = os.path.join(dataset_path, person, filename)
            if not os.path.exists(filepath):
                print(f"【警告】ファイル未存在のためスキップ: {filepath}")
                continue

            # CSV読み込み
            data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]

            # 不要列の削除（重心など：7,8,16,17,25,26,34,35,43,44,52,53）
            cap_data = np.delete(data, [7,8,16,17,25,26,34,35,43,44,52,53], axis=1)

            # choice_parts と delete_parts に応じて削除
            if choice_parts is not None and delete_parts is not None:
                to_delete = []
                for dp in delete_parts:
                    to_delete.extend(range(dp * 7, dp * 7 + 7))
                cap_choice_data = np.delete(cap_data, to_delete, axis=1)
            else:
                # 何もしない場合は cap_data をそのまま使う
                cap_choice_data = cap_data

            # data_frames 個ずつ切り出してラベル付け
            n_frames = cap_choice_data.shape[0]
            data_n = n_frames - data_frames
            for f in range(data_n):
                seq = cap_choice_data[f : f + data_frames]
                sequences.append(seq)
                labels.append(motion_idx)

    sequences = np.array(sequences)
    labels = np.array(labels)
    return sequences, labels


# --------------------------------------------------------------
# 3. 「再学習用データ」「テスト用データ」を読み込む
# --------------------------------------------------------------

# 再学習データの読み込み
fine_X, fine_Y = load_person_data(
    person=fine_tuning_person,
    days=fine_tuning_days,
    dataset_path=dataset_path,
    motions=motions,
    data_frames=data_frames,
    all_data_frames=all_data_frames,
    data_cols=data_cols,
    choice_parts=choice_parts,
    delete_parts=delete_parts
)
print(f"再学習用データ数: {len(fine_Y)}")

# テストデータの読み込み
test_X, test_Y = load_person_data(
    person=test_person,
    days=test_days,
    dataset_path=dataset_path,
    motions=motions,
    data_frames=data_frames,
    all_data_frames=all_data_frames,
    data_cols=data_cols,
    choice_parts=choice_parts,
    delete_parts=delete_parts
)
print(f"テスト用データ数: {len(test_Y)}")


# --------------------------------------------------------------
# 4. PyTorch Dataset / DataLoader の作成
# --------------------------------------------------------------
class DatasetClass(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.labels)

# 再学習用 DataLoader
t_fine_X = torch.from_numpy(fine_X).float()
t_fine_Y = torch.from_numpy(fine_Y).long()
ds_fine = DatasetClass(t_fine_X, t_fine_Y)
dl_fine = DataLoader(ds_fine, batch_size=batch_size, shuffle=True)

# テスト用 DataLoader
t_test_X = torch.from_numpy(test_X).float()
t_test_Y = torch.from_numpy(test_Y).long()
ds_test = DatasetClass(t_test_X, t_test_Y)
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------
# 5. ネットワーク定義 (元と同じ構造) ＋ 学習済みモデルのロード
# --------------------------------------------------------------
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

# 各パラメータ (元の学習と合わせる)

num_classes = len(motions)
cap_cols = len(choice_parts) * 7  # パーツ数×7DoF
model_input_dim = data_frames * cap_cols

net = MLP4(model_input_dim, fc1, fc2, num_classes).to(device)

# 学習済みモデルを読み込み
model_path = "rawlearn.path"  # 事前学習済みモデル
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"学習済みモデルをロードしました: {model_path}")
else:
    print(f"【警告】学習済みモデルファイルが見つかりません: {model_path}")


# --------------------------------------------------------------
# 6. 損失関数・最適化手法を設定
# --------------------------------------------------------------
loss_func = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)  # 適宜 lr 調整

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    total_samples = 0
    correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc


# --------------------------------------------------------------
# 7. 評価関数 (Accuracy, F1スコア)
# --------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    """
    評価時の損失、Accuracy、F1スコアを返す
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    correct = 0
    all_preds = []
    all_labels = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = loss_func(out, y)
        total_loss += loss.item()
        total_samples += y.size(0)
        preds = out.argmax(dim=1)

        correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1


# --------------------------------------------------------------
# 8. 混合行列(Confusion Matrix)作成・表示関数
# --------------------------------------------------------------
import numpy as np

@torch.no_grad()
def evaluate_test(model, loader, motions_len):
    """
    (予測, 正解) のペアをカウントして混合行列を返す。
    chart[予測][実際] の形になるが最後に転置で (実際, 予測) の並びに直す。
    """
    chart = np.zeros((motions_len, motions_len))
    model.eval()
    for X, lab in loader:
        X, lab = X.to(device), lab.to(device)
        Y = model(X)
        for i in range(len(lab)):
            predicted = Y[i].argmax().item()
            actual = lab[i].item()
            chart[predicted][actual] += 1
    return chart.T

def display_confusion_matrix(chart, class_names):
    """
    chart: 混合行列 (実際, 予測) の 2次元配列
    class_names: クラス名(モーション名)のリスト
    """
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


# --------------------------------------------------------------
# 9. 再学習(ファインチューニング)の実行
# --------------------------------------------------------------

for epoch in range(1, fine_tune_epochs + 1):
    train_loss, train_acc = train_one_epoch(net, dl_fine)
    print(f"[FineTune Epoch {epoch}/{fine_tune_epochs}] loss: {train_loss:.4f}, acc: {train_acc:.4f}")


# --------------------------------------------------------------
# 10. テストデータで評価 (Accuracy, F1スコア & 混合行列)
# --------------------------------------------------------------
test_loss, test_acc, test_f1 = evaluate(net, dl_test)
print("==== テストデータ評価結果 ====")
print(f"Loss: {test_loss:.4f}")
print(f"Acc : {test_acc:.4f}")
print(f"F1  : {test_f1:.4f}")

# 混合行列を作成・表示
chart = evaluate_test(net, dl_test, len(motions))
display_confusion_matrix(chart, motions)


# --------------------------------------------------------------
# 11. 再学習後のモデルを保存 (必要に応じて)
# --------------------------------------------------------------
fine_tuned_model_path = "fine_tuned_model.path"
torch.save(net.state_dict(), fine_tuned_model_path)
print(f"再学習後のモデルを保存しました: {fine_tuned_model_path}")
