import numpy as np
import matplotlib.pyplot as plt

# decimal 関連
import decimal
from decimal import Decimal, getcontext

import numpy as np
import matplotlib.pyplot as plt

# decimal 関連
import decimal
from decimal import Decimal, getcontext

# 計算精度 (桁数) を設定 (例: 50桁)
getcontext().prec = 50

# パラメータは Decimal 型で定義
a  = Decimal('-3.0')
b  = Decimal('2.0')
x0 = Decimal('1.0')
T  = Decimal('1.0')

trials = 10000

# --- 試験関数 φ(x) = x ---
def phi(x):
    return x

# --- 真の解 (Decimal で計算) ---
# X_t = x0 * exp((a - 0.5 b^2)*t + b*W_t)
# t, W : np.array(float) を受け取り、内部で Decimal 化
def true_solution(t, W):
    """
    t: np.array(float), shape=(n+1,)
    W: np.array(float), shape=(n+1,)
    戻り値: [Decimal, Decimal, ..., Decimal]  (長さ n+1)
    """
    X_list = []
    for i in range(len(t)):
        t_dec = Decimal(str(t[i]))  # float -> Decimal
        W_dec = Decimal(str(W[i]))  # float -> Decimal
        exponent = (a - Decimal('0.5')*(b**2))*t_dec + b*W_dec
        x_val = x0 * decimal.getcontext().exp(exponent)
        X_list.append(x_val)
    return X_list

# alpha = a - 0.5 * b^2 (Decimal)
alpha = a - Decimal('0.5')*(b**2)

# --- Euler–Maruyama 法 (Y空間, Decimal) ---
def euler_Y(y0, alpha, b, dt, W):
    """
    y0 : Decimal (初期値 = ln(x0))
    W  : np.array(float), shape=(n_steps,)
    dt : Decimal
    戻り: リスト [Y_0, Y_1, ..., Y_n] (各要素 Decimal)
    """
    n_steps = len(W)
    # Y を Decimal で格納したいので Pythonのリストで用意
    Y = [Decimal(0)]*(n_steps+1)
    Y[0] = y0
    for k in range(n_steps):
        W_dec = Decimal(str(W[k]))  # float -> Decimal
        Y[k+1] = Y[k] + alpha*dt + b*W_dec
    return Y

# --- 標準化 (zscore) は float のまま ---
def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = x.std(axis=axis, keepdims=True)
    return (x - xmean)/xstd

# --- 逆変換: X = exp(Y),  Yは Decimal リスト ---
def Y_to_X(Y):
    X_list = []
    for val in Y:
        X_list.append(decimal.getcontext().exp(val))
    return X_list

# シミュレーション & 誤差計算
n_values=[50,80,100,200,400,600,800,1000]
mean_fins_euler_Y = []

for n in n_values:
    # dt を Decimal 化
    dt = T / Decimal(str(n))
    # t_values は float で作り、後で使う
    t_values = np.linspace(0, float(T), n+1)
    
    fin_euler_Y = []
    fin_exact_solution = []
    
    for _ in range(trials):
        # 乱数を標準化 (float)
        W_zoubun_float = zscore(np.random.randn(n)) * np.sqrt(float(dt))
        # 累積和して Brown 運動 W_full
        W_sum_float = np.cumsum(W_zoubun_float)
        W_full_float = np.concatenate(([0], W_sum_float))

        # 真の解 (Decimal)
        X_exact_list = true_solution(t_values, W_full_float)
        fin_exact_solution.append(X_exact_list[-1])  # 最終値

        # Yオイラー法
        y0_dec = decimal.getcontext().ln(x0)  # ln(1.0)=0 (Decimal)
        Y_euler_list = euler_Y(y0_dec, alpha, b, dt, W_zoubun_float)
        X_euler_list = Y_to_X(Y_euler_list)
        fin_euler_Y.append(X_euler_list[-1])
    
    # 平均を Decimal で求める
    sum_euler = Decimal(0)
    sum_exact = Decimal(0)
    for val in fin_euler_Y:
        sum_euler += val
    for val in fin_exact_solution:
        sum_exact += val

    mean_fin_euler_Y = sum_euler / Decimal(str(len(fin_euler_Y)))
    mean_fin_exact_solution = sum_exact / Decimal(str(len(fin_exact_solution)))

    # 絶対誤差
    print(f'{mean_fin_euler_Y:.90f}')
    print(f'{mean_fin_exact_solution:.60f}')
    err_dec = abs(mean_fin_euler_Y - mean_fin_exact_solution)
    # グラフ用に float 化
    mean_fins_euler_Y.append(float(err_dec))

# プロット
plt.figure(figsize=(8, 6))
plt.loglog(n_values, mean_fins_euler_Y, 'o-', color='red', label='new')

# 参照線 -1
scaling_factor_euler_Y = mean_fins_euler_Y[0] / (n_values[0]**(-1.0))
n_array = np.array(n_values, dtype=float)
plt.loglog(n_array,
           scaling_factor_euler_Y * n_array**(-1.0),
           'b--', label='-1 (Euler_Y)')

plt.xlabel('Number of time steps')
plt.ylabel('Average Error')
plt.legend()
plt.grid(False)
plt.axvline(x=50)
plt.axvline(x=100)
plt.axvline(x=1000)
plt.show()
