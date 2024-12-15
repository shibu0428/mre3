import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# 混同行列のデータ
data = np.array([
    [50, 2, 1],
    [3, 45, 2],
    [4, 1, 40]
])

# クラスラベル
class_names = ['Class A', 'Class B', 'Class C']

# Pandas DataFrame に変換
df_cm = pd.DataFrame(data, index=class_names, columns=class_names)

# 図のサイズを設定
plt.figure(figsize=(8, 6))

# ヒートマップを描画
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')

# 軸ラベルとタイトルを設定
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix Heatmap', fontsize=15)

# レイアウトを自動調整
plt.tight_layout()

# 表示

plt.figure(figsize=(8, 6))
plt.imshow(df_cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix Heatmap', fontsize=15)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.show()
