import numpy as np
import matplotlib.pyplot as plt

import time

import decimal
from decimal import Decimal, getcontext

# 計算精度を設定
getcontext().prec = 150

# 乱数シード (float 乱数には影響)
np.random.seed(57)
start=time.time()

a   = Decimal('1')
b   = Decimal('0.000000000000000000001')
x0  = Decimal('1')
T   = Decimal('0.5')

trials = 1000

# 試験関数
def phi(x):
    return x

# 真の解 (Decimal)
def true_solution(t, W):
    """
    t, W は np.array(float) を仮定
    ここでは最小限の修正として "内部の演算" を Decimal 化
    """
    # 戻り値: np.array ではなく Pythonリスト (Decimalの列)
    X_list = []
    for i in range(len(t)):
        t_dec = Decimal(str(t[i]))
        W_dec = Decimal(str(W[i]))
        exponent = (a - Decimal('0.5')*(b**2)) * t_dec + b * W_dec
        X_val = x0 * decimal.getcontext().exp(exponent)
        X_list.append(X_val)
    return X_list

alpha = a - Decimal('0.5')*(b**2)

# Euler–Maruyama 法 (Y空間, Decimal)
def euler_Y(y0, alpha, b, dt, W):
    """
    y0, alpha, b, dt は Decimal
    W は np.array(float)
    """
    n_steps = len(W)
    # Y は Pythonリストで Decimal を格納
    Y = [Decimal(0)]*(n_steps+1)
    Y[0] = y0
    for k in range(n_steps):
        W_dec = Decimal(str(W[k]))
        Y[k+1] = Y[k] + alpha*dt + b*W_dec
    return Y

# 標準化 (zscore) は float のまま
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean)/(xstd)
    return zscore

# 逆変換: X = exp(Y)  (Decimal)
def Y_to_X(Y):
    X_list = []
    for val in Y:
        X_list.append(decimal.getcontext().exp(val))
    return X_list

def fin_exact_solution(trials, T, n):
    """
    内部で zscore を使った float 乱数を生成し、
    それを Decimal で厳密解に適用 (true_solution)
    のサンプル平均を取る
    """
    dt = T / Decimal(str(n))
    print(f'{dt:.60f}')
    # t_values は float でOK→ 後で Decimal 化
    t_values_float = np.linspace(0, float(T), n+1)
    exact_solutions = []
    for i in range(trials):
        W_zoubun_float = zscore(np.random.randn(n))*np.sqrt(float(dt))
        W_sum_float = np.cumsum(W_zoubun_float)
        W_full_float = np.concatenate(([0], W_sum_float))
        # Decimal で真の解
        X_list_dec = true_solution(t_values_float, W_full_float)
        exact_solutions.append(X_list_dec[-1])  # 最終値
        
    print(n, np.mean(W_zoubun_float))
    # Decimal の平均
    sum_dec = Decimal(0)
    for val in exact_solutions:
        sum_dec += val
    mean_exact_solution = sum_dec / Decimal(str(len(exact_solutions)))
    return mean_exact_solution

n_values = [10,20,30,40,60,80,100]
mean_fins_euler_Y = []

mean_exact_solution = fin_exact_solution(1000, T, 100)

# シミュレーションと誤差計算
for n in n_values:
    print(n)
    dt = T / Decimal(str(n))
    # t_values は float で作る (最小限の修正)
    t_values_float = np.linspace(0, float(T), n+1)
    print(f'{dt.sqrt():.60f}')
    fin_euler_Y = []

    for trial in range(trials):
        # float 乱数
        W_zoubun_float = Decimal((zscore(np.random.randn(n)))) * dt.sqrt()

        # Euler–Maruyama 法 (Y空間, Decimal)
        y0_dec = decimal.getcontext().ln(x0)  # ln(100) 
        Y_euler_list = euler_Y(y0_dec, alpha, b, dt, W_zoubun_float)
        # 逆変換
        X_euler_list = Y_to_X(Y_euler_list)
        fin_euler_Y.append(X_euler_list[-1])  # 最終値を保存

    # 平均 (Decimal)
    sum_dec = Decimal(0)
    for val in fin_euler_Y:
        sum_dec += val
    mean_fin_euler_Y = sum_dec / Decimal(str(len(fin_euler_Y)))
    print(f'{mean_fin_euler_Y:.60f}')
    # 誤差
    err = abs(mean_fin_euler_Y - mean_exact_solution)
    mean_fins_euler_Y.append(float(err))  # float化してプロット

print(time.time()-start)

plt.figure(figsize=(8, 6))
plt.loglog(n_values, mean_fins_euler_Y, 'o-', label='new', color='red')

# 参照線: -1 に合わせたい場合 (下記は適宜変更)
scaling_factor_euler_Y = mean_fins_euler_Y[0] / (n_values[0]**(1.0))
n_array = np.array(n_values)
plt.loglog(n_array, scaling_factor_euler_Y * n_array**(1.0), 'b--', label='-1 (Euler_Y)')

plt.xlabel('Number of time steps')
plt.ylabel('Average Error')
plt.legend()
plt.grid(False)
plt.axvline(x=1000)
plt.axvline(x=5000)
plt.axvline(x=10000)
plt.show()