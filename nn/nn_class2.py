import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import csv
from sklearn.metrics import f1_score

sns.set()

def train_and_evaluate(
    train_names,
    test_names,
    model_save_name,
    confusion_matrix_save_name,
    learning_curve_save_name
):
    #---------------------------------------------------
    # 固定パラメータや設定
    dataset_path = "../dataset_name/"
    datasetdays = ["1", "2","3","4","5"]

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

    # 全データ（すべての人）のリスト
    name_list = [
        "gou",
        "haya",
        "sibu",
        "oga",
        "yama"
    ]

    # モデルを保存するかどうか（常に保存）
    model_save = True

    # 学習時に使うフレーム数
    data_frames = 10
    # 元データから読み取る最大フレーム数
    all_data_frames = 580 + data_frames

    # バッチサイズ
    bs = 20

    # FC層のノード数
    fc1 = 1024 * 2
    fc2 = 1024 * 2

    # エポック数
    nepoch = 15

    # 取得するパーツのインデックス（不要なものはdelete_partsに入れる）
    choice_parts = [0, 1, 2, 3, 4, 5]
    delete_parts = []

    # CSVの列数 (7DoF+重心2列)*6パーツ = (7 + 2)*6 = 54
    data_cols = (7 + 2) * 6
    # キャプチャデータの列数 (7DoF)*6パーツ = 7*6 = 42
    cap_cols = 7 * 6

    # デバイスの準備
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device)
    print("CUDA 使用可能:", torch.cuda.is_available())

    # データ読み込み用リスト
    data_list = []
    labels_list = []
    person_labels_list = []

    name_to_index = {name: idx for idx, name in enumerate(name_list)}

    # ------------------------------------------------------
    # データの読み込み
    # ------------------------------------------------------
    for name in name_list:
        for date in datasetdays:
            for motion_idx, motion in enumerate(motions):
                filename = f"{motion}_{name}_{date}.csv"
                filepath = os.path.join(dataset_path, name, filename)
                print(f"読み込み中: {filepath}")

                if not os.path.exists(filepath):
                    print(f"警告: ファイルが存在しません -> {filepath}")
                    continue

                data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]

                # 不要列(重心など)を削除
                cap_data = np.delete(data, [7, 8, 16, 17, 25, 26, 34, 35, 43, 44, 52, 53], 1)

                # 選択したパーツ以外を削除
                cap_cols = len(choice_parts) * 7
                delete_list = []
                for i in delete_parts:
                    delete_list.extend(range(i * 7, i * 7 + 7))
                cap_choice_data = np.delete(cap_data, delete_list, 1)

                # データ・ラベルをリストに格納
                data_list.append(cap_choice_data)
                labels_list.append(motion_idx)
                person_labels_list.append(name_to_index[name])

    # ------------------------------------------------------
    # 学習データとテストデータを「人」で分割
    # ------------------------------------------------------
    train_person_indices = [name_to_index[n] for n in train_names]
    test_person_indices = [name_to_index[n] for n in test_names]

    train_data_sequences = []
    train_labels = []
    test_data_sequences = []
    test_labels = []

    for data_idx, data in enumerate(data_list):
        n_frames = data.shape[0]
        data_n = int(n_frames - data_frames)
        motion_label = labels_list[data_idx]
        person_label = person_labels_list[data_idx]

        if person_label in train_person_indices:
            for f in range(data_n):
                sequence = data[f : f + data_frames]
                train_data_sequences.append(sequence)
                train_labels.append(motion_label)
        elif person_label in test_person_indices:
            for f in range(data_n):
                sequence = data[f : f + data_frames]
                test_data_sequences.append(sequence)
                test_labels.append(motion_label)
        else:
            pass

    np_data = np.array(train_data_sequences)
    np_data_label = np.array(train_labels)
    np_Tdata = np.array(test_data_sequences)
    np_Tdata_label = np.array(test_labels)

    # ------------------------------------------------------
    # テンソル変換
    # ------------------------------------------------------
    t_data = torch.from_numpy(np_data)
    t_data_label = torch.from_numpy(np_data_label)
    t_Tdata = torch.from_numpy(np_Tdata)
    t_Tdata_label = torch.from_numpy(np_Tdata_label)

    # ------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------
    class dataset_class(Dataset):
        def __init__(self, data, labels):
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

    # ------------------------------------------------------
    # 1 Epoch の学習
    # ------------------------------------------------------
    def train_one_epoch(model, lossFunc, optimizer, dl):
        model.train()
        loss_sum = 0.0
        ncorrect = 0
        n = 0
        for i, (X, lab) in enumerate(dl):
            lab = lab.long()
            X, lab = X.to(device), lab.to(device)
            X = X.float()
            optimizer.zero_grad()

            Y = model(X)
            loss = lossFunc(Y, lab)
            loss.backward()
            optimizer.step()

            n += len(X)
            loss_sum += loss.item()
            ncorrect += (Y.argmax(dim=1) == lab).sum().item()
        return loss_sum / n, ncorrect / n

    # ------------------------------------------------------
    # 検証
    # ------------------------------------------------------
    @torch.no_grad()
    def evaluate(model, lossFunc, dl):
        model.eval()
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

    # ------------------------------------------------------
    # テストデータでの混同行列
    # ------------------------------------------------------
    @torch.no_grad()
    def evaluate_test(model, dl, motions_len):
        model.eval()
        chart = np.zeros([motions_len, motions_len])
        for X, lab in dl:
            lab = lab.long()
            X, lab = X.to(device), lab.to(device)
            X = X.float()
            Y = model(X)
            for i in range(len(lab)):
                predicted = Y[i].argmax().item()
                actual = lab[i].item()
                chart[predicted][actual] += 1
        return chart.T

    # ------------------------------------------------------
    # F1スコア計算
    # ------------------------------------------------------
    def calculate_f1(model, dl):
        model.eval()
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

    # ------------------------------------------------------
    # ネットワークモデル定義
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # モデル・最適化
    # ------------------------------------------------------
    net = MLP4(data_frames * cap_cols, fc1, fc2, len(motions)).to(device)
    print(net)

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # ------------------------------------------------------
    # 学習ループ
    # ------------------------------------------------------
    results = []
    print('# epoch  lossL  lossT  rateL  rateT')
    for epoch in range(1, nepoch + 1):
        lossL, rateL = train_one_epoch(net, loss_func, optimizer, dlL)
        lossT, rateT = evaluate(net, loss_func, dlT)
        results.append([epoch, lossL, lossT, rateL, rateT])

        print(f'{epoch:3d}   {lossL:.6f}   {lossT:.6f}   {rateL:.5f}   {rateT:.5f}')

    # ------------------------------------------------------
    # テストデータ評価
    # ------------------------------------------------------
    chart = evaluate_test(net, dlT, len(motions))
    f1_val = calculate_f1(net, dlT)
    print("==== Confusion Matrix (Raw) ====")
    print(chart)
    print(f"F1 Score (weighted): {f1_val:.4f}")

    # ------------------------------------------------------
    # 最後に図を作成 (混同行列 + 損失グラフ + 正解率グラフ) を1枚に
    # ------------------------------------------------------
    def plot_final_figure(conf_matrix, class_names, train_history, model, dlL, dlT, save_path, person_name):
        """
        混同行列と損失&正解率の学習曲線を1枚に出力する。
        person_name: グラフタイトルに表示する名前など
        """
        # train_history: [ [epoch, lossL, lossT, rateL, rateT], ... ]
        data = np.array(train_history)
        epochs = data[:, 0]
        train_loss = data[:, 1]
        test_loss = data[:, 2]
        train_acc  = data[:, 3]
        test_acc   = data[:, 4]

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # --- (1) 混同行列のヒートマップ (ax[0]) ---
        df_cm = pd.DataFrame(conf_matrix.astype(int), index=class_names, columns=class_names)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        # ラベル回転調整
        ax[0].set_xticklabels(class_names, rotation=45, ha='right')
        ax[0].set_yticklabels(class_names, rotation=0)

        # --- (2) Loss グラフ (ax[1]) ---
        ax[1].plot(epochs, train_loss, '.-', label='Train Loss')
        ax[1].plot(epochs, test_loss, '.-', label='Test Loss')
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        ax[1].grid(True)

        # --- (3) Accuracy グラフ (ax[2]) ---
        ax[2].plot(epochs, train_acc, '.-', label='Train Acc')
        ax[2].plot(epochs, test_acc, '.-', label='Test Acc')
        ax[2].set_title("Accuracy")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")
        ax[2].legend()
        ax[2].grid(True)

        # 全体タイトル (人名などを入れる)
        fig.suptitle(f"Result for {person_name}", fontsize=16)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"最終結果を1枚にまとめた画像として保存しました: {save_path}")

    # グラフ作成&保存 (test_names[0] の名前をタイトルに入れる例)
    plot_final_figure(
        chart,
        motions,
        results,
        net,
        dlL,
        dlT,
        save_path=learning_curve_save_name,
        person_name=test_names[0]  # タイトルに表示したい名前
    )

    # ------------------------------------------------------
    # モデルの保存
    # ------------------------------------------------------
    if model_save:
        torch.save(net.state_dict(), model_save_name)
        print(f"学習済みモデルを保存しました: {model_save_name}")
