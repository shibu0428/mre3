# fine_main.py

from testclass import test as fine_tune

# 保存用パス・ファイル名などの共通設定
out_num='1'

dir = '../fine_models3/'
models_dir='../models2/'
model_path='.path'

cm_name = '_fine.png'
lc_name = '_fine.png'

# 事前学習済みモデル（追加学習のベースとなるモデル）のパス


# --------------------------------------------------------------------------------
# 例1: train_name="yama", 学習に使う日付=["1","2","3"], テストに使う日付=["4","5"]
# --------------------------------------------------------------------------------


def train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list):
    path=test_list[0]+'_fine.path'
    name='yama'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 追加学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        fine_tune_epochs=epoc,             # エポック数
        model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
        model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
        confusion_matrix_save_name=dir + name + test_list[0] + cm_name,  # 混同行列画像の保存名
        learning_curve_save_name=dir + name + test_list[0] + lc_name      # 学習曲線画像の保存名
    )

    name='sibu'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 追加学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        fine_tune_epochs=epoc,             # エポック数
        model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
        model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
        confusion_matrix_save_name=dir + name + test_list[0] + cm_name,  # 混同行列画像の保存名
        learning_curve_save_name=dir + name + test_list[0] + lc_name      # 学習曲線画像の保存名
    )

    name='haya'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 追加学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        fine_tune_epochs=epoc,             # エポック数
        model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
        model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
        confusion_matrix_save_name=dir + name + test_list[0] + cm_name,  # 混同行列画像の保存名
        learning_curve_save_name=dir + name + test_list[0] + lc_name      # 学習曲線画像の保存名
    )

    name='gou'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 追加学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        fine_tune_epochs=epoc,             # エポック数
        model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
        model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
        confusion_matrix_save_name=dir + name + test_list[0] + cm_name,  # 混同行列画像の保存名
        learning_curve_save_name=dir + name + test_list[0] + lc_name      # 学習曲線画像の保存名
    )

    name='oga'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 追加学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        fine_tune_epochs=epoc,             # エポック数
        model_path=models_dir+name+model_path,      # 事前学習済みモデルをロード
        model_save_name=dir + name + path,            # 上書き先(または新規)のモデル保存名
        confusion_matrix_save_name=dir + name + test_list[0] + cm_name,  # 混同行列画像の保存名
        learning_curve_save_name=dir + name + test_list[0] + lc_name      # 学習曲線画像の保存名
    )

epoc=8

train_list=["2", "3", "4","5"]
test_list=["1"]
train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "3", "4","5"]
test_list=["2"]
train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "2", "4","5"]
test_list=["3"]
train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "2", "3","5"]
test_list=["4"]
train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1","2", "3", "4"]
test_list=["5"]
train_fc_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)