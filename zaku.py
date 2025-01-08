import numpy as np
import matplotlib.pyplot as plt


a = -1.0  
b = 2.0   
x0 = 1.0  
T = 1.0   
trial = 20000

# 試験関数
def phi(x):
    return x**2


def true_solution(t, W):
    return x0*np.exp((a-0.5*b**2)*t+b*W)



# オイラー法
def euler(x0, a, b, dt, W):
    n_steps=len(W)
    X=np.zeros(n_steps + 1)
    X[0]=x0
    for i in range(n_steps):
        X[i+1]=X[i]+a*X[i]*dt+b*X[i]*W[i]
    return X

#ミルシュタイン法
def milstein(x0, a, b, dt, W):
    n_steps=len(W)
    X=np.zeros(n_steps + 1)
    X[0]=x0
    for i in range(n_steps):
        X[i+1]=X[i]+a*X[i]*dt+b*X[i]*W[i]+0.5*b**2*X[i]*(W[i]**2-dt)
    return X


n_values=[10, 20, 40, 80, 100, 200, 400, 600, 800, 1000]
n_values = np.logspace(np.log10(10), np.log10(100000), num=20, dtype=int)
n_values = sorted(set(n_values))  # 重複を避けてソート
#n_values=range(10,100,50)
mean_fins_euler = []
mean_fins_milstein = []






# シミュレーションと誤差計算
for n in n_values:
    dt=T/n
    t_values=np.linspace(0, T, n+1)
    

    fin_euler = []   #確率過程の最後
    fin_milstein = []
    fin_exact_solution = []
    
    for trial in range(trial):
        #同じWを使用
        W_zoubun = np.random.randn(n) *np.sqrt(dt)
        W_sum = np.cumsum(W_zoubun)
        W_full = np.concatenate(([0], W_sum))  # ここで初期値0を追加
        #真の解
        exact_solution = true_solution(t_values, W_full) 
        fin_exact_solution.append(phi(exact_solution[-1]))
        #オイラー法
        X_euler = euler(x0, a, b, dt, W_zoubun)
        fin_euler.append(phi(X_euler[-1]))
        
        #ミルシュタイン法
        X_milstein = milstein(x0, a, b, dt, W_zoubun)
        fin_milstein.append(phi(X_milstein[-1]))
    #print(n,len(W_his),np.mean(W_his))
    # 平均誤差を計算
    mean_fin_euler = np.mean(fin_euler)
    print(f"{n:4},{mean_fin_euler}")
    mean_fin_milstein = np.mean(fin_milstein)
    mean_fin_exact_solution = np.mean(fin_exact_solution)
    mean_fins_euler.append(np.abs(mean_fin_euler-mean_fin_exact_solution))
    mean_fins_milstein.append(np.abs(mean_fin_milstein-mean_fin_exact_solution))
print("mean_fin_exs",mean_fin_exact_solution)
plt.figure(figsize=(8, 6))

plt.loglog(n_values, mean_fins_euler, 'o-', label='Euler-Maruyama', color='purple')
plt.loglog(n_values, mean_fins_milstein, 's-', label='Milstein', color='green')

# 理論的な収束率の調整 
scaling_factor_euler = mean_fins_euler[0] / (n_values[0]**(-1.0))
scaling_factor_milstein = mean_fins_milstein[0] / (n_values[0]**(-2.0))


n_array = np.array(n_values)
plt.loglog(n_array, scaling_factor_euler * n_array**(-1.0), 'y--', label='-1/2 (Euler)')
plt.loglog(n_array, scaling_factor_milstein * n_array**(-2.0), 'c--', label='-1 (Milstein)')


plt.xlabel('Number of time steps')
plt.ylabel('Average Maximum Error')
plt.legend()


plt.grid(False)
plt.axvline(x=10)
plt.axvline(x=100)
plt.axvline(x=1000)
plt.show()
