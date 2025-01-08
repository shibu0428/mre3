import numpy as np
import matplotlib.pyplot as plt


a = 0.01
b = 0.01
x0 = np.exp(10)
T = 0.001
trials = 1000
np.set_printoptions(precision=30)
# 試験関数
def phi(x):
    return x**0.1


def true_solution(t, W):
    return x0*np.exp((a-0.5*b**2)*t+b*W)

alpha = a - 0.5*(b**2)

# Euler–Maruyama 法 (Y空間)
def euler_Y(y0, alpha, b, dt, W):
    n_steps = len(W)
    Y = np.zeros(n_steps+1)
    Y[0] = 10
    for k in range(n_steps):
        Y[k+1] = Y[k] + alpha*dt + b*W[k]
    return Y

#標準化
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


# 逆変換: X = exp(Y)
def Y_to_X(Y):
    return np.exp(Y)


n_values=[10,20,50,80,100,200,300, 400,500, 600,700, 800,900, 1000,1200,1400,1600,1700,1800,1900,2000,4000,5000,6000,7000,8000,9000,10000]                               #200, 400, 600, 800, 1000,1500,3000,6000,10000
#n_values = np.logspace(np.log10(10), np.log10(100000), num=20, dtype=int)
#n_values = sorted(set(n_values))  # 重複を避けてソート
#n_values=range(10,100,50)
mean_fins_euler_Y = []





# シミュレーションと誤差計算
for n in n_values:
    #print(n,"start")
    dt=T/n
    t_values=np.linspace(0, T, n+1)
    
    fin_euler_Y = []
    fin_exact_solution = []
    
    for trial in range(trials):
        #同じWを使用
        W_zoubun=zscore(np.random.randn(n))*np.sqrt(dt)
        #W_zoubun=np.random.randn(n)*np.sqrt(dt)
        W_sum = np.cumsum(W_zoubun)
        W_full = np.concatenate(([0], W_sum))  # ここで初期値0を追加
        #真の解
        exact_solution = true_solution(t_values, W_full) 
        fin_exact_solution.append(phi(exact_solution[-1]))
        #Yオイラー法
        Y_euler = euler_Y(np.log(x0), alpha, b, dt, W_zoubun)
        fin_euler_Y.append(phi(Y_to_X(Y_euler[-1])))

    mean_fin_euler_Y = np.mean(fin_euler_Y)
    mean_fin_exact_solution = np.mean(fin_exact_solution)
    print(f'{mean_fin_euler_Y:.90f}')
    print(f'{mean_fin_exact_solution:.60f}')

    mean_fins_euler_Y.append(np.abs(mean_fin_euler_Y-mean_fin_exact_solution))
    
    print(f'{mean_fins_euler_Y[-1]:.60f}')
#print("mean_fin_exs",mean_fin_exact_solution)
plt.figure(figsize=(8, 6))
plt.loglog(n_values, mean_fins_euler_Y, 'r-', label='new', color='red')
# 理論的な収束率の調整 

scaling_factor_euler_Y = mean_fins_euler_Y[0] / (n_values[0]**(-1.0))

n_array = np.array(n_values)
print(mean_fins_euler_Y)
plt.loglog(n_array, scaling_factor_euler_Y * n_array**(-1.0), 'b--', label='-1 (Euler_Y)')

plt.xlabel('Number of time steps')
plt.ylabel('Average Error')
plt.legend()


plt.grid(False)
plt.axvline(x=50)
plt.axvline(x=100)
plt.axvline(x=1000)
plt.show()