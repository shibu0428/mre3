import numpy as np
import matplotlib.pyplot as plt

# 初期条件と時間区間の設定
x0 = 1.0
T  = 1.0

# ---- 時間依存の関数 ----
def a(t):
    return t

def b(t):
    return np.sqrt(t)

# Euler-Maruyama法の実装（離散的な増分を使用）
def euler_method_discrete(x0, dt, W):
    n = len(W)
    X = np.zeros(n+1)
    X[0] = x0
    for k in range(n):
        t_k = k * dt
        X[k+1] = X[k] + a(t_k)*X[k]*dt + b(t_k)*X[k]*W[k]
    return X

# 離散的なWiener増分の生成
def discrete_W(n, dt):
    # 増分を +sqrt(dt) または -sqrt(dt) からランダムに選択
    return np.random.choice([np.sqrt(dt), -np.sqrt(dt)], size=n)

# 真の期待値の計算
def true_expectation(T):
    return np.exp(T**2 / 2)

# 弱収束性のシミュレーションと評価
def simulate_weak_convergence(n_values, trials):
    e_true = true_expectation(T)  # 真の期待値

    # 弱収束性の誤差記録用
    weak_errors_discrete = []

    for n in n_values:
        print(f"ステップ数 n = {n}")
        dt = T / n
        t_values = np.linspace(0, T, n+1)

        # 各試行の最終時刻 X_T を記録
        final_X = np.zeros(trials)

        for trial in range(trials):
            # 離散的なWiener増分を生成
            dW_discrete = discrete_W(n, dt)
            
            # Euler-Maruyama法でシミュレーション
            X_discrete = euler_method_discrete(x0, dt, dW_discrete)
            
            # 最終時刻 T の値を記録
            final_X[trial] = X_discrete[-1]
        
        # サンプル平均を計算
        mean_X_discrete = np.mean(final_X)
        
        # 相対弱誤差を計算
        weak_error = abs(mean_X_discrete - e_true) 
        weak_errors_discrete.append(weak_error)
        
        print(f"Relative Weak Error: {weak_error:.7f}")

    # プロット
    plt.figure(figsize=(8,6))
    plt.loglog(n_values, weak_errors_discrete, 's-', color='orange', label='Weak Error (Discrete Increments)')

    # 理論的な弱収束率 O(Δt) を目安として補助線を引く
    # Δt = T / n, よって誤差 ∝ n^{-1}
    scaling_factor = weak_errors_discrete[0] * (n_values[0]**1)
    n_array = np.array(n_values)
    plt.loglog(n_array, scaling_factor * n_array**(-1.0), 'y--', label='Slope -1 (O(Δt))')

    plt.xlabel('Number of time steps (n)')
    plt.ylabel('Relative Weak Error')
    plt.title('Weak Convergence of Euler-Maruyama with Discrete Increments')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# ----------------------------
# 実行例 (main)
# ----------------------------
if __name__ == "__main__":
    n_values = [10, 50, 100, 300, 500, 700, 1000, 2000,3000,4000]
    trials = 30000
    simulate_weak_convergence(n_values, trials)