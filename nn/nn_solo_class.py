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

def load_person_data(
    person, file_nums, dataset_path, motions,
    data_frames=10, all_data_frames=590,
    data_cols=54,
    choice_parts=None, delete_parts=None
):
    """
    単一の人物 (person) が複数のファイル番号 (file_nums) を持つデータを読み込み、
    data_frames個ずつシーケンス化して返す。
    """
    sequences = []
    labels = []

    for num in file_nums:
        for motion_idx, motion in enumerate(motions):
            filename = f"{motion}_{person}_{num}.csv"
            filepath = os.path.join(dataset_path, person, filename)
            if not os.path.exists(filepath):
                print(f"【警告】ファイル未存在のためスキップ: {filepath}")
                continue

            # CSV読み込み
            data = np.genfromtxt(filepath, delimiter=',', filling_values=0)[:all_data_frames, :data_cols]

            # 不要列の削除（重心など）
            cap_data = np.delete(
                data, [7, 8, 16, 17, 25, 26, 34, 35, 43, 44, 52, 53],
                axis=1
            )

            # choice_parts と delete_parts を使って削除
            if choice_parts is not None and delete_parts is not None:
                to_delete = []
                for dp in delete_parts:
                    to_delete.extend(range(dp * 7, dp * 7 + 7))
                cap_choice_data = np.delete(cap_data, to_delete, axis=1)
            else:
                cap_choice_data = cap_data

            # data_frames 個ずつ切り出して1サンプルとする
            n_frames = cap_choice_data.shape[0]
            data_n = n_frames - data_frames
            for f in range(data_n):
                seq = cap_choice_data[f : f + data_frames]
                sequences.append(seq)
                labels.append(motion_idx)

    return np.array(sequences), np.array(labels)


class MLP4(nn.Module):
    """
    本研究で用いる MLP (Multi-Layer Perceptron) の例
    """
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


def train_single_person(
    person_name,
    train_nums,            # 4つの番号 (例: ["1", "2", "3", "4"])
    test_num,              # 1つの番号 (例: ["5"]) ※リスト形式でもOK
    model_save_name,
    output_image_name
):
    """
    単一被験者が持つ5個のデータセットのうち4つを学習( train_nums )に,
    残り1つ( test_num )をテストに使用し,
    最終的にモデルを保存 & 混同行列+学習曲線を1枚の画像に出力する。

    Parameters
    ----------
    person_name : str
        被験者名 (例: "yama")
    train_nums : list of str
        学習用に使用するファイル番号 (4つ)
    test_num : list of str
        テスト用に使用するファイル番号 (1つ)
    model_save_name : str
        保存する学習済みモデルのファイル名
    output_image_name : str
        混同行列と学習曲線をまとめた画像ファイル名
    """

    # ------------------------
    # 1. パラメータ設定
    # ------------------------
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

    choice_parts = [0, 1, 2, 3, 4, 5]
    delete_parts = []

    cap_cols = len(choice_parts) * 7
    model_input_dim = data_frames * cap_cols
    num_classes = len(motions)

    fc1 = 1024 * 2
    fc2 = 1024 * 2

    batch_size = 20
    nepoch = 15  # エポック数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------
    # 2. データ読み込み
    # ------------------------
    # 学習用 (4つのファイル番号)
    train_X, train_Y = load_person_data(
        person=person_name,
        file_nums=train_nums,
        dataset_path=dataset_path,
        motions=motions,
        data_frames=data_frames,
        all_data_frames=all_data_frames,
        data_cols=data_cols,
        choice_parts=choice_parts,
        delete_parts=delete_parts
    )
    print(f"学習用データ数: {len(train_Y)}")

    # テスト用 (1つのファイル番号)
    # ※ 引数 test_num が1つなら ["5"], 複数なら ["5","..."] になる想定
    test_X, test_Y = load_person_data(
        person=person_name,
        file_nums=test_num,
        dataset_path=dataset_path,
        motions=motions,
        data_frames=data_frames,
        all_data_frames=all_data_frames,
        data_cols=data_cols,
        choice_parts=choice_parts,
        delete_parts=delete_parts
    )
    print(f"テスト用データ数: {len(test_Y)}")

    # numpy -> tensor
    t_train_X = torch.from_numpy(train_X).float()
    t_train_Y = torch.from_numpy(train_Y).long()
    t_test_X = torch.from_numpy(test_X).float()
    t_test_Y = torch.from_numpy(test_Y).long()

    # ------------------------
    # 3. Dataset / DataLoader
    # ------------------------
    class dataset_class(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        def __len__(self):
            return len(self.labels)

    ds_train = dataset_class(t_train_X, t_train_Y)
    ds_test  = dataset_class(t_test_X, t_test_Y)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

    print(f"DL_train: {len(ds_train)}, DL_test: {len(ds_test)}")

    # ------------------------
    # 4. モデル構築
    # ------------------------
    net = MLP4(model_input_dim, fc1, fc2, num_classes).to(device)
    print(net)

    # 損失関数
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    # 最適化
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # ------------------------
    # 5. 学習＆評価に使う関数
    # ------------------------
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
        avg_acc  = correct / total_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        all_true = []
        all_pred = []
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_func(out, y)

            total_loss += loss.item()
            total_samples += y.size(0)

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc  = correct / total_samples
        f1 = f1_score(all_true, all_pred, average='weighted')
        return avg_loss, avg_acc, f1

    @torch.no_grad()
    def evaluate_confusion(model, loader, motions_len):
        # 混同行列作成
        chart = np.zeros([motions_len, motions_len])
        model.eval()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            for i in range(len(y)):
                pred_idx = out[i].argmax().item()
                actual = y[i].item()
                chart[pred_idx][actual] += 1
        return chart.T

    # ------------------------
    # 6. 学習ループ
    # ------------------------
    results = []  # [ (epoch, train_loss, test_loss, train_acc, test_acc), ... ]
    print("\n# epoch | train_loss | test_loss | train_acc | test_acc")
    for epoch in range(1, nepoch + 1):
        tr_loss, tr_acc = train_one_epoch(net, dl_train)
        te_loss, te_acc, _ = evaluate(net, dl_test)
        results.append([epoch, tr_loss, te_loss, tr_acc, te_acc])
        print(f"{epoch:3d}    {tr_loss:.6f}    {te_loss:.6f}    {tr_acc:.4f}    {te_acc:.4f}")

    # テスト最終評価
    final_loss, final_acc, final_f1 = evaluate(net, dl_test)
    print("\n=== テスト最終評価 ===")
    print(f"Loss: {final_loss:.4f}")
    print(f"Acc : {final_acc:.4f}")
    print(f"F1  : {final_f1:.4f}")

    # 混同行列
    conf_matrix = evaluate_confusion(net, dl_test, num_classes)

    hikensha_key={
        "sibu":"A",
        "yama":"B",
        "haya":"C",
        "gou":"D",
        "oga":"E",
    }
    hikensha=hikensha_key.get(train_name,"ERROR")
    
    # ------------------------
    # 7. 混同行列 + 学習曲線の描画
    # ------------------------
    def plot_final_figure(conf_matrix, class_names, history, person_name, save_path):
        data = np.array(history)
        epochs = data[:, 0]
        train_loss = data[:, 1]
        test_loss  = data[:, 2]
        train_acc  = data[:, 3]
        test_acc   = data[:, 4]

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # (1) 混同行列
        df_cm = pd.DataFrame(conf_matrix.astype(int), index=class_names, columns=class_names)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        ax[0].set_xticklabels(class_names, rotation=45, ha='right')
        ax[0].set_yticklabels(class_names, rotation=0)

        # (2) Loss
        ax[1].plot(epochs, train_loss, '.-', label='Train Loss')
        ax[1].plot(epochs, test_loss, '.-', label='Test Loss')
        ax[1].set_title("Loss Curve")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].set_ylim(0, 6)     # 損失関数を 0~6 に固定
        ax[1].legend()
        ax[1].grid(True)

        # (3) Accuracy
        ax[2].plot(epochs, train_acc, '.-', label='Train Acc')
        ax[2].plot(epochs, test_acc, '.-', label='Test Acc')
        ax[2].set_title("Accuracy Curve")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")
        ax[2].set_ylim(0, 1.0)   # 正解率を 0~1 に固定
        ax[2].legend()
        ax[2].grid(True)

        fig.suptitle(f"Training Result for {person_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"\n混同行列 + 学習曲線を1枚にまとめた画像を保存しました: {save_path}")

    # 描画・保存
    plot_final_figure(conf_matrix, motions, results, hikensha+test_num[0], output_image_name)

    # ------------------------
    # 8. モデル保存
    # ------------------------
    torch.save(net.state_dict(), model_save_name)
    print(f"モデルを保存しました: {model_save_name}")
