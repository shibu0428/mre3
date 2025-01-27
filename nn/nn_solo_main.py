# fine_main.py

from nn_solo_class import train_single_person

# 保存用パス・ファイル名などの共通設定
out_num='1'

dir = '../fine_models3/'
models_dir='../models2/'
model_path='.path'

cm_name = '_fine.png'
lc_name = '_fine.png'

# 事前学習済みモデル（追加学習のベースとなるモデル）のパス
    """
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

# --------------------------------------------------------------------------------
# 例1: train_name="yama", 学習に使う日付=["1","2","3"], テストに使う日付=["4","5"]
# --------------------------------------------------------------------------------


def train(name,train_list,test_list):
    path=test_list[0]+'.path'
    fine_tune(
        train_name=name,
        train_num=train_list,        # 学習に使うファイル番号
        test_num=test_list,             # テストで使うファイル番号
        model_save_name=dir + name+test_list[0] + path,            # モデル保存名
        output_image_name=dir + name + test_list[0] + ".png",  # 混同行列画像の保存名
    )

def train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list):
    path=test_list[0]+'_fine.path'
    name='yama'
    train(name,train_list,test_list)

    name='sibu'
    train(name,train_list,test_list)


    name='haya'
    train(name,train_list,test_list)


    name='gou'
    train(name,train_list,test_list)

    name='oga'
    train(name,train_list,test_list)
epoc=999
train_list=["2", "3", "4","5"]
test_list=["1"]
train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "3", "4","5"]
test_list=["2"]
train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "2", "4","5"]
test_list=["3"]
train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1", "2", "3","5"]
test_list=["4"]
train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)

train_list=["1","2", "3", "4"]
test_list=["5"]
train_member(dir,models_dir,model_path,cm_name,lc_name,epoc,train_list,test_list)