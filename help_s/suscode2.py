def true_solution():
        X="近似解"
    return X


def euler_Y(y0, dt, W_incr):
    n = len(W_incr)
    Y = np.zeros(n+1)
    Y[0] = y0
    for k in range(n):
        t_k = k * dt
        Y[k+1] = Y[k] + alpha_func(t_k)*dt + b(t_k)*W_incr[k]
    return Y


# 1) 真の解 (離散近似) を計算
X_true_path = true_solution()
for n in n_values:
    dt = T / n
    t_values = "0~Tまでのn+1個の等間隔な数列"
    
    fin_euler_Y  = []
    fin_exact    = []
    
    for _ in range(trials):
        # Wiener 過程の増分を作成
        dW ="正規分布に従う乱数配列"*"sqrt(dt)"

        # 4) Euler in Y-space
        y0 = np.log(x0)
        Y_euler_path = euler_Y(y0, dt, dW)
        X_eulerY_path = Y_to_X(Y_euler_path)
        
        # 終了時刻 T の値
        fin_exact.append(X_true_path[-1])
        fin_euler_Y.append(X_eulerY_path[-1])
    
    # 各手法の「最終時刻 T における X_T」のサンプル平均
    mean_fin_exact     = np.mean(fin_exact)
    mean_fin_eulerY    = np.mean(fin_euler_Y)
    
    # 真の解の期待値 (サンプル平均) との絶対誤差を弱収束的に評価
    err_euler     = abs(mean_fin_euler - mean_fin_exact) / abs(mean_fin_exact)
    err_eulerY    = abs(mean_fin_eulerY - mean_fin_exact) / abs(mean_fin_exact)
    
    mean_fins_euler.append(err_euler)
    mean_fins_eulerY.append(err_eulerY)