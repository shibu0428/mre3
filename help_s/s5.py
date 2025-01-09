import numpy as np
import matplotlib.pyplot as plt

# ===== Decimal 設定 =====
import decimal
from decimal import Decimal, getcontext

getcontext().prec = 50  # 必要に応じて桁数を調整

# 乱数シード (float の乱数には影響)
np.random.seed(57)

# ===== パラメータを Decimal で定義 =====
a   = Decimal('0.1')
b   = Decimal('0.0')
x0  = Decimal('100')
T   = Decimal('0.5')

trials = 1000

# --- 試験関数 ---
def phi(x):
    return x

# --- 真の解 (Decimal) ---
def true_solution(t, W):
    """
    t, W : np.array(float) (最小限の修正)
    ループで Decimal に変換して指数計算
    戻り値: [Decimal,...]
    """
    X_list = []
    for i in range(len(t)):
        t_dec = Decimal(str(t[i]))
        W_dec = Decimal(str(W[i]))
        exponent = (a - Decimal('0.5')*(b**2)) * t_dec + b * W_dec
        X_val = x0 * decimal.getcontext().exp(exponent)
        X_list.append(X_val)
    return X_list

alpha = a - Decimal('0.5')*(b**2)

# --- Euler–Maruyama 法 (Y空間, Decimal) ---
def euler_Y(y0, alpha, b, dt, W):
    """
    y0, alpha, b, dt は Decimal
    W は [float] だが、中で Decimal 変換
    戻り値: [Decimal, Decimal, ...] (Y_0..Y_n)
    """
    n_steps = len(W)
    Y = [Decimal(0)]*(n_steps+1)
    Y[0] = y0
    for k in range(n_steps):
        W_dec = Decimal(str(W[k]))
        Y[k+1] = Y[k] + alpha*dt + b*W_dec
    return Y

# --- 標準化 (zscore) を Decimal で行う ---
def zscore(x_list):
    """
    x_list: List[Decimal]
    戻り値: List[Decimal]
    """
    n_len = Decimal(len(x_list))
    # 平均
    mean_val = sum(x_list)/n_len
    #print(f'{mean_val:.120f}')
    # 分散 = (1/n)*Σ(xi - mean)^2
    # 標準偏差 = sqrt(分散)
    var_val = Decimal('0')
    for xi in x_list:
        diff = xi - mean_val
        var_val += diff*diff
    var_val /= n_len
    std_val = var_val.sqrt()  # Decimalのsqrt
    # (x - mean)/std
    z_list = []
    for xi in x_list:
        z_list.append( (xi - mean_val)/std_val )
    return z_list

# --- 逆変換: X = exp(Y) (Decimal) ---
def Y_to_X(Y):
    X_list = []
    for val in Y:
        X_list.append(decimal.getcontext().exp(val))
    return X_list

def fin_exact_solution(trials, T, n):
    """
    Decimal化した「真の解の最終値」サンプル平均を返す
    ここでも zscore を Decimal で実行
    """
    dt = T / Decimal(str(n))
    t_values_float = np.linspace(0, float(T), n+1)  # float
    exact_solutions = []

    for i in range(trials):
        # (1) float 乱数を生成
        w_float = np.random.randn(n)
        # (2) Decimal のリストに変換
        w_dec_list = [Decimal(str(x)) for x in w_float]
        # (3) zscore(Decimal) -> 平均0, 分散1
        w_dec_z = zscore(w_dec_list)
        # (4) dt も Decimal
        dt_sqrt = dt.sqrt()  # √dt
        # (5) W_zoubun = zscore(...) * √dt
        w_zoubun_dec = [z * dt_sqrt for z in w_dec_z]

        # 累積和
        W_full_dec = [Decimal('0')]
        cum = Decimal('0')
        for dw in w_zoubun_dec:
            cum += dw
            W_full_dec.append(cum)

        # Decimalの真の解 (最終値)
        X_list_dec = []
        for i_t in range(len(t_values_float)):
            t_dec = Decimal(str(t_values_float[i_t]))
            exponent = (a - Decimal('0.5')*(b**2)) * t_dec + b*W_full_dec[i_t]
            x_val = x0 * decimal.getcontext().exp(exponent)
            X_list_dec.append(x_val)

        exact_solutions.append(X_list_dec[-1])

    # 平均
    sum_dec = Decimal('0')
    for val in exact_solutions:
        sum_dec += val
    mean_exact_solution = sum_dec / Decimal(str(len(exact_solutions)))
    return mean_exact_solution

n_values = [200, 500, 1000, 4000, 8000, 20000]
mean_fins_euler_Y = []

# 真の解推定
mean_exact_solution = fin_exact_solution(1000, T, 100)

# シミュレーションと誤差計算
for n in n_values:
    dt = T / Decimal(str(n))

    fin_euler_Y = []
    for trial in range(trials):
        # float → Decimal
        w_float = np.random.randn(n)
        w_dec_list = [Decimal(str(x)) for x in w_float]
        # zscore(Decimal)
        w_dec_z = zscore(w_dec_list)
        dt_sqrt = dt.sqrt()
        # W_zoubun = zscore(...) * √dt (Decimal)
        w_zoubun_dec = [z * dt_sqrt for z in w_dec_z]

        # Euler–Maruyama 法 (Y空間, Decimal)
        y0_dec = decimal.getcontext().ln(x0)  # ln(100)
        Y_euler_list = [Decimal(0)]*(n+1)
        Y_euler_list[0] = y0_dec
        for k in range(n):
            Y_euler_list[k+1] = Y_euler_list[k] + alpha*dt + b*w_zoubun_dec[k]

        # 逆変換
        X_euler_list = []
        for val in Y_euler_list:
            X_euler_list.append(decimal.getcontext().exp(val))
        fin_euler_Y.append(X_euler_list[-1])

    # fin_euler_Y の平均 (Decimal)
    sum_dec = Decimal('0')
    for val in fin_euler_Y:
        sum_dec += val
    mean_fin_euler_Y = sum_dec / Decimal(str(len(fin_euler_Y)))

    err_dec = abs(mean_fin_euler_Y - mean_exact_solution)
    mean_fins_euler_Y.append(float(err_dec))  # グラフ用に float化

plt.figure(figsize=(8, 6))
plt.loglog(n_values, mean_fins_euler_Y, 'o-', label='new', color='red')

# 参照線 (例: -1)
import math
scaling_factor_euler_Y = mean_fins_euler_Y[0] / (n_values[0]**(1.0))
n_array = np.array(n_values, dtype=float)
plt.loglog(n_array, scaling_factor_euler_Y * n_array**(1.0), 'b--', label='slope=-1')

plt.xlabel('Number of time steps')
plt.ylabel('Average Error (Decimal, zscore in Decimal)')
plt.legend()
plt.grid(False)
plt.show()
