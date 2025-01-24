from nn_class import train_and_evaluate

dir='../models/'
path='.path'
cm_name='cm.png'
lc_name='lc.png'


train_names = ["sibu", "haya", "oga", "gou"]  # 学習に使う人
test_names  = ["yama"]                       # テストに使う人

train_and_evaluate(
    train_names,
    test_names,
    model_save_name=dir+test_names[0]+path,                # 学習結果のモデルファイル名
    confusion_matrix_save_name=dir+test_names[0]+cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir+test_names[0]+lc_name       # 学習曲線画像の保存名
)

train_names = ["yama", "haya", "oga", "gou"]  # 学習に使う人
test_names  = ["sibu"]                      # テストに使う人

train_and_evaluate(
    train_names,
    test_names,
    model_save_name=dir+test_names[0]+path,                # 学習結果のモデルファイル名
    confusion_matrix_save_name=dir+test_names[0]+cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir+test_names[0]+lc_name       # 学習曲線画像の保存名
)

train_names = ["sibu", "yama", "oga", "gou"]  # 学習に使う人
test_names  = ["haya"]                       # テストに使う人

train_and_evaluate(
    train_names,
    test_names,
    model_save_name=dir+test_names[0]+path,                # 学習結果のモデルファイル名
    confusion_matrix_save_name=dir+test_names[0]+cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir+test_names[0]+lc_name       # 学習曲線画像の保存名
)

train_names = ["sibu", "haya", "yama", "gou"]  # 学習に使う人
test_names  = ["oga"]                       # テストに使う人

train_and_evaluate(
    train_names,
    test_names,
    model_save_name=dir+test_names[0]+path,                # 学習結果のモデルファイル名
    confusion_matrix_save_name=dir+test_names[0]+cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir+test_names[0]+lc_name       # 学習曲線画像の保存名
)

train_names = ["sibu", "haya", "oga", "yama"]  # 学習に使う人
test_names  = ["gou"]                       # テストに使う人

train_and_evaluate(
    train_names,
    test_names,
    model_save_name=dir+test_names[0]+path,                # 学習結果のモデルファイル名
    confusion_matrix_save_name=dir+test_names[0]+cm_name,  # 混同行列画像の保存名
    learning_curve_save_name=dir+test_names[0]+lc_name       # 学習曲線画像の保存名
)