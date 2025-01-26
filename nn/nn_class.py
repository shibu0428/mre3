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

    # モデルを保存するかどうか（この実装では常に保存する形にしています）
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

    # CUDAデバイスの準備
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device)
    print("CUDA 使用可能:", torch.cuda.is_available())

    # データ格納用のリスト
    data_list = []
    labels_list = []
    person_labels_list = []  # 人のラベル（名前 -> index）

    # 人の名前をインデックス化する辞書
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
                    continue  # ファイルが存在しない場合はスキップ

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
            # 学習データに追加
            for f in range(data_n):
                sequence = data[f : f + data_frames]
                train_data_sequences.append(sequence)
                train_labels.append(motion_label)

        elif person_label in test_person_indices:
            # テストデータに追加
            for f in range(data_n):
                sequence = data[f : f + data_frames]
                test_data_sequences.append(sequence)
                test_labels.append(motion_label)
        else:
            # いずれにも該当しない場合は無視（実装によっては変更しても良い）
            pass

    # numpy 配列化
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
    # データセット / データローダー定義
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
    # 混同行列の表示＆保存関数
    # ------------------------------------------------------
    def display_confusion_matrix(chart, class_names, save_path=None):
        # pandas DataFrame に変換
        df_cm = pd.DataFrame(chart.astype(int), index=class_names, columns=class_names)

        print("Confusion Matrix:")
        print(df_cm)

        # ヒートマップ描画
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')

        if save_path is not None:
            plt.savefig(save_path)
            print(f"混同行列を画像として保存しました: {save_path}")

        plt.close()  # 表示を閉じる

    # ------------------------------------------------------
    # 1 Epoch の学習を行う関数
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
    # 検証（損失と正解率の計算）
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
    # テストデータでの混同行列を作成
    # ------------------------------------------------------
    def evaluate_test(model, dl, motions_len):
        model.eval()
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

    # ------------------------------------------------------
    # 学習経過を可視化する関数 (学習曲線)
    # ------------------------------------------------------
    def plot_learning_curve(results, model, loss_func, dlL, dlT, m_size, subtitle, save_path=None):
        data = np.array(results)
        fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(12, 4))

        # Loss
        ax[0].plot(data[:, 0], data[:, 1], '.-', label='training data')
        ax[0].plot(data[:, 0], data[:, 2], '.-', label='test data')
        ax[0].axhline(0.0, color='gray')
        ax[0].set_ylim(-0.05, 1.75)
        ax[0].legend()
        ax[0].set_title('loss')

        # Accuracy
        ax[1].plot(data[:, 0], data[:, 3], '.-', label='training data')
        ax[1].plot(data[:, 0], data[:, 4], '.-', label='test data')
        ax[1].axhline(1.0, color='gray')
        ax[1].set_ylim(0.35, 1.01)
        ax[1].legend()
        ax[1].set_title('accuracy')

        fig.suptitle('modelSize' + str(m_size) + str(subtitle))

        # 学習後の損失と識別率の表示
        loss2, rrate = evaluate(model, loss_func, dlL)
        print(f'# [最終] 学習データ: 損失 {loss2:.5f}  識別率 {rrate:.4f}')
        loss2, rrate = evaluate(model, loss_func, dlT)
        print(f'# [最終] テストデータ: 損失 {loss2:.5f}  識別率 {rrate:.4f}')

        if save_path is not None:
            plt.savefig(save_path)
            print(f"学習曲線を画像として保存しました: {save_path}")

        plt.close()

    # ------------------------------------------------------
    # F1スコアを計算する関数
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
    # モデルおよび最適化設定
    # ------------------------------------------------------
    net = MLP4(data_frames * cap_cols, fc1, fc2, len(motions)).to(device)
    print(net)

    # 損失関数（交差エントロピー）
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    # パラメータ最適化器
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # ------------------------------------------------------
    # 学習ループ
    # ------------------------------------------------------
    results = []
    print('# epoch  lossL  lossT  rateL  rateT')
    for t in range(1, nepoch + 1):
        lossL, rateL = train_one_epoch(net, loss_func, optimizer, dlL)
        lossT, rateT = evaluate(net, loss_func, dlT)
        results.append([t, lossL, lossT, rateL, rateT])

        if (t % 1 == 0):
            print(f'{t:3d}   {lossL:.6f}   {lossT:.6f}   {rateL:.5f}   {rateT:.5f}')

    # ------------------------------------------------------
    # テストデータでの混同行列を作成して表示・保存
    # ------------------------------------------------------
    chart = evaluate_test(net, dlT, len(motions))
    print(chart)
    display_confusion_matrix(chart, motions, save_path=confusion_matrix_save_name)

    # 学習曲線の描画・保存
    plot_learning_curve(results, net, loss_func, dlL, dlT, [fc1, fc2], "_final", save_path=learning_curve_save_name)

    # F1スコアの計算・表示
    f1_val = calculate_f1(net, dlT)
    print(f"F1 Score (weighted): {f1_val:.4f}")

    # ------------------------------------------------------
    # モデルの保存
    # ------------------------------------------------------
    if model_save:
        torch.save(net.state_dict(), model_save_name)
        print(f"学習済みモデルを保存しました: {model_save_name}")
