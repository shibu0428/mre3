# fine_tune_model.py (学習せずテストのみ行う版)

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


def test(
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
    もともとは追加学習(ファインチューニング)を行う関数でしたが、
    ここでは「学習を行わずテストだけ」を実行するように改変しています。

    引数・呼び出し方法は元コードと互換性を保ちつつ、train_num などの指定があっても
    学習は行わず、既存モデルを読み込んでテストを行うだけにしています。

    Parameters
    ----------
    train_name : str
        学習に使用する特定の1人の名前 (今回は学習には使用しないが引数は残す)
    train_num : list of str
        学習に使用するファイル番号のリスト (同上)
    test_num : list of str
        テストに使用するファイル番号のリスト
    fine_tune_epochs : int
        追加学習のエポック数 (学習しないので無視される)
    model_path : str
        事前学習済みモデルのパス
    model_save_name : str
        本来は追加学習後に保存するモデルのパス (今回は保存しないが引数は残す)
    confusion_matrix_save_name : str
        (元のコードに合わせて残すが、実際にはこのファイル名では保存していない)
    learning_curve_save_name : str
        混同行列を保存するときに使用するファイル名
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
    # 2. 「学習データ (train_num)」「テストデータ (test_num)」の読み込み
    #    ※ 今回は学習しないが、互換性のため train_num 部分も読み込むだけ実装
    # --------------------------------------------------------------
    # （train_num データを実際には学習に使わないが、一応ロードしておく）
    '''
    _fine_X, _fine_Y = load_person_data(
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
    '''
    #print(f"元コード上では再学習用データ数 (train) : {len(_fine_Y)} (今回は学習に使用しません)")

    # テストデータの読み込み
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
    #print(f"テスト用データ数 (test) : {len(test_Y)}")

    # --------------------------------------------------------------
    # 3. Dataset / DataLoader の作成 (テスト用)
    # --------------------------------------------------------------
    class DatasetClass(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        def __len__(self):
            return len(self.labels)

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
        #print(f"事前学習済みモデルを読み込みました: {model_path}")
    else:
        print(f"【警告】モデルファイル {model_path} が見つかりません。")

    # --------------------------------------------------------------
    # 5. 損失関数・評価関数など
    #    (学習は行わないため、optimizer は用意しない)
    # --------------------------------------------------------------
    loss_func = nn.CrossEntropyLoss(reduction='sum')

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
    # 6. (学習は行わず) テストデータでの評価のみ行う
    # --------------------------------------------------------------
    final_test_loss, final_test_acc, final_test_f1 = evaluate(net, dl_test)
    #print("==== テストデータ評価結果 ====")
    #print(f"Loss: {final_test_loss:.4f}")
    print(f"{train_name},{test_num[0]} Acc : {final_test_acc:.4f}")
    #print(f"F1  : {final_test_f1:.4f}")

    # 混同行列作成
    chart = evaluate_confusion(net, dl_test, len(motions))

    # --------------------------------------------------------------
    # 7. 結果の可視化 (混同行列のみを保存)
    #    学習曲線はないため、混同行列だけをプロットします
    # --------------------------------------------------------------
    hikensha_key = {
        "sibu":"A",
        "yama":"B",
        "haya":"C",
        "gou":"D",
        "oga":"E",
    }
    hikensha = hikensha_key.get(train_name, "ERROR")

    def plot_confusion_matrix(conf_matrix, class_names, title_name, save_path):
        """
        混同行列だけを1枚に描画・保存
        """
        df_cm = pd.DataFrame(conf_matrix.astype(int), index=class_names, columns=class_names)
        plt.figure(figsize=(6,6))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix - {title_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"混同行列を画像として保存しました: {save_path}")

    # 混同行列を出力
    '''
    plot_confusion_matrix(
        chart,
        motions,
        f"{hikensha} / TestNum={test_num}",
        learning_curve_save_name  # 元引数を使い回し
    )
    '''

    # --------------------------------------------------------------
    # 8. 追加学習後のモデル保存 (今回は学習していないため無効化)
    #    下記の通り、学習結果を保存しないが、呼び出しエラーを防ぐため残す
    # --------------------------------------------------------------
    # torch.save(net.state_dict(), model_save_name)
    # print(f"再学習無しですが、引数上のモデル保存先: {model_save_name}")
    #print("【INFO】学習は行っていないためモデルの上書き保存は行いません。")
