# fine_main.py

from fine_tune_class import fine_tune

# 保存用パス・ファイル名などの共通設定
out_num='1'

#dir = '../fine_models/'
dir='../nosave/'
models_dir='../models/'
model_path='.path'
path = out_num+'_fine.path'
cm_name = out_num+'_finecm.png'
lc_name = out_num+'_finelc.png'

# 事前学習済みモデル（追加学習のベースとなるモデル）のパス


# --------------------------------------------------------------------------------
# 例1: train_name="yama", 学習に使う日付=["1","2","3"], テストに使う日付=["4","5"]
# --------------------------------------------------------------------------------

train_list=["5"]
test_list=["1"]
epoc=8

name='haya'

out_num=test_list[0]
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

test_list=["2"]
out_num=test_list[0]

dir='../nosave/'
models_dir='../models/'
model_path='.path'
path = out_num+'_fine.path'
cm_name = out_num+'_finecm.png'
lc_name = out_num+'_finelc.png'

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

test_list=["3"]
out_num=test_list[0]

dir='../nosave/'
models_dir='../models/'
model_path='.path'
path = out_num+'_fine.path'
cm_name = out_num+'_finecm.png'
lc_name = out_num+'_finelc.png'

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

test_list=["4"]
out_num=test_list[0]

dir='../nosave/'
models_dir='../models/'
model_path='.path'
path = out_num+'_fine.path'
cm_name = out_num+'_finecm.png'
lc_name = out_num+'_finelc.png'

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
