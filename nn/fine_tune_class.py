# fine_tune_model.py

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
# データを読み込み、(data_frames 個ずつ) シーケンス化する関数
# --------------------------------------------------------------
def load_person_data(
    person, file_nums, dataset_path, motions,
    data_frames=10, all_data_frames=590,
    data_cols=54,
    choice_parts=None, delete_parts=None
):
    """
    特定の person と複数ファイル番号 (file_nums) からモーションCSVを読み込み、
    data_frames個ずつ切り出したシーケンスとラベルを返す。
    """
    sequences = []
    labels = []

    for num in file_nums:
        for motion_idx, motion in enumerate(motions):
            # CSVファイル名の組み立て (例: motion_person_num.csv)
            filename = f"{motion}_{person}_{num}.csv"
            filepath = os.path.join(dataset_path, person, filename)
            if not os.path.exists(filepath):
                print(f"【警告】ファイル未存在のためスキップ: {filepath}")
                continue

            # CSV読み込み
            data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]

            # 不要列の削除（重心など：7,8,16,17,25,26,34,35,43,44,52,53）
            cap_data = np.delete(
                data, [7, 8, 16, 17, 25, 26, 34, 35, 43, 44, 52, 53],
                axis=1
            )

            # choice_parts と delete_parts に応じて削除
            if choice_parts is not None and delete_parts is not None:
                to_delete = []
                for dp in delete_parts:
                    to_delete.extend(range(dp * 7, dp * 7 + 7))
                cap_choice_data = np.delete(cap_data, to_delete, axis=1)
            else:
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
# ネットワーク定義 (元のMLP構造)
# --------------------------------------------------------------
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


def fine_tune(
    train_name,
    train_num,
    test_num,
    fine_tune_epochs,
    model_path,
    model_save_name,
    confusion_matrix_save_name,
    learning_curve_save_name
):
    """
    追加学習(ファインチューニング)を行う関数。

    Parameters
    ----------
    train_name : str
        学習に使用する特定の1人の名前
    train_num : list of str
        学習に使用するファイル番号のリスト (例: ["1", "2", "3"])
    test_num : list of str
        テストに使用するファイル番号のリスト (例: ["4", "5"])
    fine_tune_epochs : int
        追加学習のエポック数
    model_path : str
        事前学習済みモデルのパス
    model_save_name : str
        追加学習後に保存するモデルのパス
    confusion_matrix_save_name : str
        混同行列を画像として保存するファイル名
    learning_curve_save_name : str
        学習曲線を画像として保存するファイル名
    """

    # --------------------------------------------------------------
    # 1. 固定パラメータや設定（必要に応じて書き換えてください）
    # --------------------------------------------------------------
    dataset_path = "../dataset_name/"

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
        # "golf"
    ]

    data_frames = 10
    all_data_frames = 580 + data_frames
    data_cols = (7 + 2) * 6

    # 選択するパーツ (各パーツ7DoF) -> 下記のように必要に応じて設定
    choice_parts = [0, 1, 2, 3, 4, 5]
    delete_parts = []  # 削除パーツがあれば指定

    cap_cols = len(choice_parts) * 7  # パーツ数×7DoF
    model_input_dim = data_frames * cap_cols
    num_classes = len(motions)

    fc1 = 1024 * 2
    fc2 = 1024 * 2

    batch_size = 20

    # デバイス選択
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    # --------------------------------------------------------------
    # 2. 「学習データ」「テストデータ」の読み込み
    # --------------------------------------------------------------
    fine_X, fine_Y = load_person_data(
        person=train_name,
        file_nums=train_num,
        dataset_path=dataset_path,
        motions=motions,
        data_frames=data_frames,
        all_data_frames=all_data_frames,
        data_cols=data_cols,
        choice_parts=choice_parts,
        delete_parts=delete_parts
    )
    print(f"再学習用データ数 (train) : {len(fine_Y)}")

    test_X, test_Y = load_person_data(
        person=train_name,   # 同じ person だがファイル番号が異なる
        file_nums=test_num,
        dataset_path=dataset_path,
        motions=motions,
        data_frames=data_frames,
        all_data_frames=all_data_frames,
        data_cols=data_cols,
        choice_parts=choice_parts,
        delete_parts=delete_parts
    )
    print(f"テスト用データ数 (test) : {len(test_Y)}")

    # --------------------------------------------------------------
    # 3. Dataset / DataLoader の作成
    # --------------------------------------------------------------
    class DatasetClass(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        def __len__(self):
            return len(self.labels)

    t_fine_X = torch.from_numpy(fine_X).float()
    t_fine_Y = torch.from_numpy(fine_Y).long()
    ds_fine = DatasetClass(t_fine_X, t_fine_Y)
    dl_fine = DataLoader(ds_fine, batch_size=batch_size, shuffle=True)

    t_test_X = torch.from_numpy(test_X).float()
    t_test_Y = torch.from_numpy(test_Y).long()
    ds_test = DatasetClass(t_test_X, t_test_Y)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------------
    # 4. モデルの定義とロード
    # --------------------------------------------------------------
    net = MLP4(model_input_dim, fc1, fc2, num_classes).to(device)

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        #print(f"学習済みモデルをロードしました: {model_path}")
    else:
        print(f"【警告】モデルファイル {model_path} が見つかりません。")

    # --------------------------------------------------------------
    # 5. 損失関数・最適化手法
    # --------------------------------------------------------------
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)  # 必要に応じて学習率を変更

    # --------------------------------------------------------------
    # 学習用の関数
    # --------------------------------------------------------------
    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0
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

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()

        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0
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

    # 混同行列作成
    @torch.no_grad()
    def evaluate_test(model, loader, motions_len):
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

    def display_confusion_matrix(chart, class_names, save_path=None):
        df_cm = pd.DataFrame(chart.astype(int), index=class_names, columns=class_names)

        #print("Confusion Matrix:")
        #print(df_cm)

        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')

        if save_path is not None:
            plt.savefig(save_path)
            #print(f"混同行列を画像として保存しました: {save_path}")
        plt.close()

    # 学習曲線描画用 (epochごとに保存した結果をまとめる)
    training_history = []  # [(epoch, train_loss, train_acc, test_loss, test_acc), ...]

    # --------------------------------------------------------------
    # 6. 追加学習ループ
    # --------------------------------------------------------------
    for epoch in range(1, fine_tune_epochs + 1):
        train_loss, train_acc = train_one_epoch(net, dl_fine)
        test_loss, test_acc, _ = evaluate(net, dl_test)

        # 学習履歴を保存
        training_history.append((epoch, train_loss, train_acc, test_loss, test_acc))

        print(f"[FineTune Epoch {epoch}/{fine_tune_epochs}] "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    # --------------------------------------------------------------
    # 7. テストデータでの最終評価 + 混同行列
    # --------------------------------------------------------------
    final_test_loss, final_test_acc, final_test_f1 = evaluate(net, dl_test)
    print("==== テストデータ評価結果 (最終) ====")
    print(f"Loss: {final_test_loss:.4f}")
    print(f"Acc : {final_test_acc:.4f}")
    print(f"F1  : {final_test_f1:.4f}")

    # 混同行列の作成・表示・保存
    chart = evaluate_test(net, dl_test, len(motions))
    display_confusion_matrix(chart, motions, confusion_matrix_save_name)

    # --------------------------------------------------------------
    # 8. 学習曲線を描画して保存
    # --------------------------------------------------------------
    # training_history = [(epoch, train_loss, train_acc, test_loss, test_acc), ...]
    epochs = [row[0] for row in training_history]
    train_losses = [row[1] for row in training_history]
    train_accs   = [row[2] for row in training_history]
    test_losses  = [row[3] for row in training_history]
    test_accs    = [row[4] for row in training_history]

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test  Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, test_accs, label='Test  Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.suptitle('Fine Tuning Learning Curve')

    plt.tight_layout()
    plt.savefig(learning_curve_save_name)
    #print(f"学習曲線を画像として保存しました: {learning_curve_save_name}")
    plt.close()

    # --------------------------------------------------------------
    # 9. 再学習後のモデルを保存
    # --------------------------------------------------------------
    torch.save(net.state_dict(), model_save_name)
    #print(f"再学習後のモデルを保存しました: {model_save_name}")
