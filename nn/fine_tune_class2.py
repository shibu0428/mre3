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
import time

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
            # CSVファイル名の組み立て
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

            # choice_parts と delete_parts に応じた削除
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
    追加学習(ファインチューニング)を行い、混同行列と学習曲線を
    1枚の画像にまとめて出力する。

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
        (不要になるが引数の互換のため残置)
    learning_curve_save_name : str
        (最終的に混同行列 + 学習曲線を1つにまとめて保存するファイル名)
    """

    # --------------------------------------------------------------
    # 1. 固定パラメータや設定
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
    ]

    data_frames = 10
    all_data_frames = 580 + data_frames
    data_cols = (7 + 2) * 6

    # 選択パーツ
    choice_parts = [0, 1, 2, 3, 4, 5]
    delete_parts = []

    cap_cols = len(choice_parts) * 7
    model_input_dim = data_frames * cap_cols
    num_classes = len(motions)

    fc1 = 1024 * 2
    fc2 = 1024 * 2

    batch_size = 20

    # デバイス選択
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    else:
        print(f"【警告】モデルファイル {model_path} が見つかりません。")

    # --------------------------------------------------------------
    # 5. 損失関数・最適化手法
    # --------------------------------------------------------------
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # --------------------------------------------------------------
    # 学習関数 / 評価関数 定義
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

    @torch.no_grad()
    def evaluate_confusion(model, loader, motions_len):
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

    # --------------------------------------------------------------
    # 6. 追加学習ループ
    # --------------------------------------------------------------
    training_history = []  # [(epoch, train_loss, train_acc, test_loss, test_acc), ...]
    starttime=time.time()
    for epoch in range(1, fine_tune_epochs + 1):
        train_loss, train_acc = train_one_epoch(net, dl_fine)
        test_loss, test_acc, _ = evaluate(net, dl_test)

        training_history.append((epoch, train_loss, train_acc, test_loss, test_acc))

        print(f"[FineTune {epoch}/{fine_tune_epochs}] "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")
    endtime=time.time()
    # --------------------------------------------------------------
    # 7. テストデータでの評価
    # --------------------------------------------------------------
    final_test_loss, final_test_acc, final_test_f1 = evaluate(net, dl_test)
    print("==== テストデータ評価結果 (最終) ====")
    print(f"time:  {endtime-starttime}")
    print(f"Loss: {final_test_loss:.4f}")
    print(f"Acc : {final_test_acc:.4f}")
    print(f"F1  : {final_test_f1:.4f}")

    chart = evaluate_confusion(net, dl_test, len(motions))

    hikensha_key={
        "sibu":"A",
        "yama":"B",
        "haya":"C",
        "gou":"D",
        "oga":"E",
    }
    hikensha=hikensha_key.get(train_name,"ERROR")

    # --------------------------------------------------------------
    # 8. 混同行列+学習曲線を1枚にまとめて保存
    # --------------------------------------------------------------
    def plot_final_figure(conf_matrix, class_names, history, hikensha, save_path):
        """
        混同行列と学習曲線を1枚にまとめて描画・保存する。
        person_name: グラフのタイトル等に使用
        """
        data = np.array(history)
        epochs = data[:, 0]
        train_loss = data[:, 1]
        train_acc  = data[:, 2]
        test_loss  = data[:, 3]
        test_acc   = data[:, 4]

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # --- (1) 混同行列 ---
        df_cm = pd.DataFrame(conf_matrix.astype(int), index=class_names, columns=class_names)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        # X/Yのラベルが見切れないよう回転
        ax[0].set_xticklabels(class_names, rotation=45, ha='right')
        ax[0].set_yticklabels(class_names, rotation=0)

        # --- (2) Loss ---
        ax[1].plot(epochs, train_loss, '.-', label='Train Loss')
        ax[1].plot(epochs, test_loss, '.-', label='Test Loss')
        ax[1].set_title("Loss Curve")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].set_ylim(0, 0.8)      # ← 縦軸のスケールを 0 から 6 に設定
        ax[1].legend()
        ax[1].grid(True)

        # --- (3) Accuracy ---
        ax[2].plot(epochs, train_acc, '.-', label='Train Acc')
        ax[2].plot(epochs, test_acc, '.-', label='Test Acc')
        ax[2].set_title("Accuracy Curve")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")
        ax[2].set_ylim(0.6, 1.0)      # ← 縦軸のスケールを 0 から 6 に設定
        ax[2].legend()
        ax[2].grid(True)

        fig.suptitle(f"Fine Tuning Result for {hikensha}", fontsize=16)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"混同行列と学習曲線を1枚にまとめた画像を保存しました: {save_path}")



    # 1枚にまとめて保存
    plot_final_figure(chart, motions, training_history, hikensha+test_num[0], learning_curve_save_name)

    # --------------------------------------------------------------
    # 9. 追加学習後のモデルを保存
    # --------------------------------------------------------------
    torch.save(net.state_dict(), model_save_name)
    print(f"再学習後のモデルを保存しました: {model_save_name}")
