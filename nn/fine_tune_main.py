# fine_main.py

from fine_tune_class import fine_tune

# 保存用パス・ファイル名などの共通設定
dir = '../fine_models/'
models_dir='../models/'
model_path='.path'
path = '_fine.path'
cm_name = '_finecm.png'
lc_name = '_finelc.png'

# 事前学習済みモデル（追加学習のベースとなるモデル）のパス


# --------------------------------------------------------------------------------
# 例1: train_name="yama", 学習に使う日付=["1","2","3"], テストに使う日付=["4","5"]
# --------------------------------------------------------------------------------

train_list=["1", "2", "4"]
test_list=["3","5"]
epoc=8

name='yama'
fine_tune(
    train_name=name,
    train_num=train_list,        # 追加学習に使うファイル番号
    test_num=test_list,             # テストで使うファイル番号
    fine_tune_epochs=epoc,             # エポック数
    model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
    model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
    confusion_matrix_save_name=dir + name + cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir + name + lc_name      # 学習曲線画像の保存名
)

name='sibu'
fine_tune(
    train_name=name,
    train_num=train_list,        # 追加学習に使うファイル番号
    test_num=test_list,             # テストで使うファイル番号
    fine_tune_epochs=epoc,             # エポック数
    model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
    model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
    confusion_matrix_save_name=dir + name + cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir + name + lc_name      # 学習曲線画像の保存名
)

name='haya'
fine_tune(
    train_name=name,
    train_num=train_list,        # 追加学習に使うファイル番号
    test_num=test_list,             # テストで使うファイル番号
    fine_tune_epochs=epoc,             # エポック数
    model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
    model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
    confusion_matrix_save_name=dir + name + cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir + name + lc_name      # 学習曲線画像の保存名
)

name='gou'
fine_tune(
    train_name=name,
    train_num=train_list,        # 追加学習に使うファイル番号
    test_num=test_list,             # テストで使うファイル番号
    fine_tune_epochs=epoc,             # エポック数
    model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
    model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
    confusion_matrix_save_name=dir + name + cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir + name + lc_name      # 学習曲線画像の保存名
)

name='oga'
fine_tune(
    train_name=name,
    train_num=train_list,        # 追加学習に使うファイル番号
    test_num=test_list,             # テストで使うファイル番号
    fine_tune_epochs=epoc,             # エポック数
    model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
    model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
    confusion_matrix_save_name=dir + name + cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir + name + lc_name      # 学習曲線画像の保存名
)