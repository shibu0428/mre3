import numpy as np
import matplotlib.pyplot as plt


x0 = 1.0
T  = 1.0


# ---- 時間依存の関数 ----
def a(t):
    return t

def b(t):
    return np.sqrt(t)


def true_solution(t, W):

    n = len(t) - 1  # ステップ数
    dt = t[1] - t[0]

    drift = 0.25 * t**2  # shape (n+1,)

  
    diffuse = np.zeros(n+1)
    for k in range(n):
        t_k = t[k]
        dWk = W[k+1] - W[k]
        diffuse[k+1] = diffuse[k] + np.sqrt(t_k)*dWk

    X = x0 * np.exp(drift + diffuse)
    return X


def euler_method(x0, dt, W):
    
    n = len(W)
    X = np.zeros(n+1)
    X[0] = x0
    for k in range(n):
        t_k = k * dt
        X[k+1] = X[k] + a(t_k)*X[k]*dt + b(t_k)*X[k]*W[k]
    return X


def milstein_method(x0, dt, W):
  
    n = len(W)
    X = np.zeros(n+1)
    X[0] = x0
    for k in range(n):
        t_k = k * dt
        incr = a(t_k)*X[k]*dt + b(t_k)*X[k]*W[k]
        # Milstein の補正項
        # b'(t_k) = d/dt sqrt(t) = 1/(2 sqrt(t))
        # => 0.5*b(t_k)*b'(t_k) = 0.5*sqrt(t_k)*(1/(2 sqrt(t_k)))= 1/4,  (t_k>0)
        # ただし t_k=0 の時は厳密には∞…
        if t_k > 0:
            milstein_correction = 0.25 * X[k] * (W[k]**2 - dt)
        else:
            milstein_correction = 0.0  # t_k=0 のステップは補正を0にする等
        X[k+1] = X[k] + incr + milstein_correction
    return X


def alpha_func(t):
    return 0.5 * t

def euler_Y(y0, dt, W_incr):
    n = len(W_incr)
    Y = np.zeros(n+1)
    Y[0] = y0
    for k in range(n):
        t_k = k * dt
        Y[k+1] = Y[k] + alpha_func(t_k)*dt + b(t_k)*W_incr[k]
    return Y

def Y_to_X(Y):
    return np.exp(Y)


def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    return (x - xmean)/xstd

def true_expectation(T):
    return np.exp(T**2 / 2)

def discrete_W(n, dt):
    # 増分を +sqrt(dt) または -sqrt(dt) からランダムに選択
    return np.random.choice([np.sqrt(dt), -np.sqrt(dt)], size=n)

def simulate_and_plot(n_values, trials):
    '''
    q=10000
    dt = T / q
    truetry=1000
    t_values = np.linspace(0, T, q+1)
    fin_exact    = []
    for tt in range(truetry):
        if tt%1000==0:
            print(tt,"/",truetry)
        dW =np.random.randn(q)*np.sqrt(dt)
            
        W_full = np.concatenate(([0.0], np.cumsum(dW)))
        X_true_path = true_solution(t_values, W_full)
        fin_exact.append(X_true_path[-1])
    mean_fin_exact     = np.mean(fin_exact)
    print(mean_fin_exact )
    '''

    e=1.64872127070012819416

    mean_fins_euler    = []
    mean_fins_milstein = []
    mean_fins_eulerY   = []
    
    for n in n_values:
        print(n)
        dt = T / n
        # 時刻配列
        t_values = np.linspace(0, T, n+1)

        fin_euler    = []
        fin_milstein = []
        fin_euler_Y  = []
        
        for num in range(trials):
            if num%100000==0:
                print(num)
            # Wiener 過程の増分を作成
            #dW = discrete_W(n, dt)
            dW =np.random.randn(n)*np.sqrt(dt)
            
            
            
            # 2) Euler
            X_euler_path = euler_method(x0, dt, dW)

            dW =np.random.randn(n)*np.sqrt(dt)

            #dW = discrete_W(n, dt)
            # 3) Milstein
            X_mil_path   = milstein_method(x0, dt, dW)

            dW =np.random.randn(n)*np.sqrt(dt)

            #dW = discrete_W(n, dt)
            # 4) Euler in Y-space
            y0 = np.log(x0)
            Y_euler_path = euler_Y(y0, dt, dW)
            X_eulerY_path = Y_to_X(Y_euler_path)
            
            # 終了時刻 T の値
            fin_euler.append(X_euler_path[-1])
            fin_milstein.append(X_mil_path[-1])
            fin_euler_Y.append(X_eulerY_path[-1])
        
        # 各手法の「最終時刻 T における X_T」のサンプル平均
        mean_fin_euler     = np.mean(fin_euler)
        mean_fin_milstein  = np.mean(fin_milstein)
        mean_fin_eulerY    = np.mean(fin_euler_Y)
        print(mean_fin_euler)
        
        # 真の解の期待値 (サンプル平均) との絶対誤差を弱収束的に評価
        err_euler     = abs(mean_fin_euler - e) /e
        err_milstein  = abs(mean_fin_milstein - e) / e
        err_eulerY    = abs(mean_fin_eulerY - e) / e
        mean_fins_euler.append(err_euler)
        mean_fins_milstein.append(err_milstein)
        mean_fins_eulerY.append(err_eulerY)
    # プロット
    plt.figure(figsize=(8,6))
    plt.loglog(n_values, mean_fins_euler,    'o-', color='purple', label='Euler-Maruyama')
    plt.loglog(n_values, mean_fins_milstein, 's-', color='green',  label='Milstein')
    plt.loglog(n_values, mean_fins_eulerY,   '^-', color='red',    label='Ishiwata-Euler')

    # 目安として slope -1 の補助線を引く (弱収束 ~1次を想定)
    scaling_factor_euler = mean_fins_eulerY[0] / (n_values[0]**(-1.0))
    n_array = np.array(n_values)
    plt.loglog(n_array, scaling_factor_euler*n_array**(-1.0), 'y--', label='slope -1')

    plt.xlabel('Number of time steps (n)')
    plt.ylabel('Relative Weak Error')
    plt.legend()
    plt.grid(True)
    plt.title('Weak Convergence')
    plt.show()


# ----------------------------
# 実行例 (main)
# ----------------------------
if __name__ == "__main__":
    n = [10,50,100,300,500,700,1000,1500,2000]
    simulate_and_plot(n, trials=2000000)
